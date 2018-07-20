extern crate alsa;
extern crate libc;
extern crate nix;

use a52_decoder;
use base::*;
use std;
use std::collections::VecDeque;
use std::error;
use std::ffi::CString;
use time::{Time, TimeDelta};

use super::input::*;

const CHANNELS: usize = 2;

const ACCEPTED_RATES: [usize; 4] = [44100, 48000, 88200, 96000];
const MAX_SAMPLE_RATE_ERROR: i32 = 1000;

struct RateDetector {
    window: TimeDelta,
    history: VecDeque<(Time, usize)>,
    sum: usize,
}

impl RateDetector {
    fn new(window: TimeDelta) -> RateDetector {
        return RateDetector {
            window: window,
            history: VecDeque::new(),
            sum: 0,
        };
    }

    fn update(&mut self, samples: usize, t_end: Time) -> usize {
        self.sum += samples;
        self.history.push_back((t_end, samples));
        while self.history.len() > 0 && t_end - self.history[0].0 >= self.window {
            self.sum -= self.history.pop_front().unwrap().1;
        }

        let period = t_end - self.history[0].0;
        if period <= TimeDelta::zero() {
            return 48000;
        }

        (self.sum - self.history[0].1) * 1000 / (period.in_milliseconds() as usize)
    }

    fn reset(&mut self) {
        self.history.clear();
        self.sum = 0;
    }
}

pub struct AlsaInput {
    spec: DeviceSpec,
    pcm: alsa::PCM,
    sample_rate: usize,
    format: SampleFormat,
    period_size: usize,
    device_delay: usize,
    rate_detector: RateDetector,
    active: bool,
    last_non_silent_time: Time,

    a52_decoder: a52_decoder::A52Decoder,
    a52_stream: bool,
}

// Deactivate input after 5 seconds of silence.
const SILENCE_PERIOD_SECONDS: i64 = 5;

impl AlsaInput {
    pub fn open(spec: DeviceSpec, period_duration: TimeDelta) -> Result<Box<AlsaInput>> {
        let pcm = try!(alsa::PCM::open(
            &*CString::new(&spec.id[..]).unwrap(),
            alsa::Direction::Capture,
            false
        ));

        let mut sample_rate = spec.sample_rate.unwrap_or(48000);
        let period_size;
        let format;
        let device_delay;
        {
            let hwp = try!(alsa::pcm::HwParams::any(&pcm));
            try!(hwp.set_channels(CHANNELS as u32));
            try!(hwp.set_rate(sample_rate as u32, alsa::ValueOr::Nearest));

            for fmt in [
                alsa::pcm::Format::S32LE,
                alsa::pcm::Format::S243LE,
                alsa::pcm::Format::S24LE,
                alsa::pcm::Format::S16LE,
            ].iter()
            {
                match hwp.set_format(*fmt) {
                    Ok(_) => break,
                    Err(_) => (),
                }
            }

            let target_period_size = period_duration * sample_rate as i64 / TimeDelta::seconds(1);
            try!(hwp.set_period_size_near(
                target_period_size as alsa::pcm::Frames,
                alsa::ValueOr::Nearest
            ));
            try!(hwp.set_periods(8, alsa::ValueOr::Nearest));
            try!(hwp.set_rate_resample(false));
            try!(hwp.set_access(alsa::pcm::Access::RWInterleaved));
            try!(pcm.hw_params(&hwp));

            sample_rate = try!(hwp.get_rate()) as usize;
            period_size = try!(hwp.get_period_size()) as usize;
            format = match try!(hwp.get_format()) {
                alsa::pcm::Format::S16LE => SampleFormat::S16LE,
                alsa::pcm::Format::S243LE => SampleFormat::S24LE3,
                alsa::pcm::Format::S24LE => SampleFormat::S24LE4,
                alsa::pcm::Format::S32LE => SampleFormat::S32LE,
                fmt => return Err(Error::new(&format!("Unknown format: {:?}", fmt))),
            };

            device_delay = (pcm.avail_delay()?).1 as usize + period_size * 2;

            println!(
                "INFO: Reading {} ({}). Buffer size: {}x{}. {} {}",
                spec.id,
                &spec.name,
                period_size,
                try!(hwp.get_periods()),
                sample_rate,
                format
            );
        }

        let now = Time::now();
        Ok(Box::new(AlsaInput {
            spec: spec,
            pcm: pcm,
            sample_rate: sample_rate,
            format: format,
            period_size: period_size,
            device_delay: device_delay,
            rate_detector: RateDetector::new(TimeDelta::milliseconds(500)),
            active: false,
            last_non_silent_time: now,
            a52_decoder: a52_decoder::A52Decoder::new(),
            a52_stream: false,
        }))
    }

    fn read_raw(&mut self) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; self.period_size * CHANNELS * self.format.bytes_per_sample()];
        let read_result = self.pcm.io().readi(&mut buf[..]);
        let samples = match read_result {
            Ok(0) => return Err(Error::new("snd_pcm_readi returned 0.")),
            Ok(r) => r,
            Err(e) => {
                println!("Recovering AlsaInput {}: {:?}", &self.spec.name, e.errno());
                self.rate_detector.reset();
                match self
                    .pcm
                    .recover(e.errno().map(|x| x as i32).unwrap_or(0), true)
                {
                    Ok(_) => return self.read_raw(),
                    Err(_) => return Err(Error::new(error::Error::description(&e))),
                }
            }
        };
        assert!(samples > 0);

        let bytes = samples * CHANNELS * self.format.bytes_per_sample();
        buf.resize(bytes, 0);

        if self.spec.enable_a52 {
            self.a52_decoder
                .add_data(&buf[..], self.format.bytes_per_sample());
        }

        Ok(buf)
    }

    fn read_pcm(&mut self) -> Result<Option<Frame>> {
        let buf = self.read_raw()?;

        // Check if we want to switch to A52.
        if self.spec.enable_a52 {
            if self.a52_decoder.have_sync() {
                self.a52_stream = true;
                return self.read_a52();
            }
        }

        let delay = (self.pcm.avail_delay()?).1;
        let now = Time::now();
        let samples = buf.len() / (CHANNELS * self.format.bytes_per_sample());
        let end_timestamp = now - samples_to_timedelta(self.sample_rate as f64, delay as i64);
        let timestamp =
            end_timestamp - samples_to_timedelta(self.sample_rate as f64, samples as i64);

        let rate = self.rate_detector.update(samples, end_timestamp);
        let mut locked_rate = 0;
        for r in &ACCEPTED_RATES[..] {
            if (rate as i32 - *r as i32).abs() < MAX_SAMPLE_RATE_ERROR {
                locked_rate = *r;
            }
        }

        if locked_rate == 0 {
            self.active = false;
            return Ok(None);
        }

        if locked_rate != self.sample_rate {
            self.sample_rate = locked_rate;
            println!(
                "INFO: rate changed for input device {}: {}",
                &self.spec.name, self.sample_rate
            );
        }

        let mut all_zeros = true;
        for &v in buf.iter() {
            if v != 0 {
                all_zeros = false;
                break;
            }
        }

        if !all_zeros {
            self.active = true;
            self.last_non_silent_time = now;
        } else if self.active
            && now - self.last_non_silent_time > TimeDelta::seconds(SILENCE_PERIOD_SECONDS)
        {
            self.active = false;
        }

        if !self.active {
            return Ok(None);
        }

        Ok(Some(Frame::from_buffer_stereo(
            self.format,
            self.sample_rate as f64,
            &buf[..],
            timestamp,
        )))
    }

    fn read_a52(&mut self) -> Result<Option<Frame>> {
        // Read data until we have a frame.
        while !self.a52_decoder.have_frame() {
            self.read_raw()?;
            if self.a52_decoder.fallback_to_pcm() {
                self.a52_stream = false;
                return Ok(None);
            }
        }

        Ok(self.a52_decoder.get_frame())
    }
}

impl Input for AlsaInput {
    fn read(&mut self) -> Result<Option<Frame>> {
        if self.a52_stream {
            self.read_a52()
        } else {
            self.read_pcm()
        }
    }

    fn min_delay(&self) -> TimeDelta {
        let mut r = samples_to_timedelta(
            self.sample_rate as f64,
            (self.device_delay + self.period_size) as i64,
        );
        if self.a52_stream {
            r += self.a52_decoder.delay()
        }
        r
    }
}

const TARGET_SAMPLE_RATES: [usize; 2] = [44100, 48000];

const RETRY_PERIOD_MS: i64 = 3000;
const PROBE_TIME_MS: i64 = 500;

pub struct ResilientAlsaInput {
    spec: DeviceSpec,
    period_duration: TimeDelta,
    input: Option<Box<AlsaInput>>,
    last_open_attempt: Time,
    probe_sample_rate: bool,
    last_active: Time,
    next_sample_rate_to_probe: usize,
}

impl ResilientAlsaInput {
    pub fn new(
        spec: DeviceSpec,
        period_duration: TimeDelta,
        probe_sample_rate: bool,
    ) -> Box<ResilientAlsaInput> {
        let now = Time::now();
        Box::new(ResilientAlsaInput {
            spec: spec,
            period_duration: period_duration,
            input: None,
            probe_sample_rate: probe_sample_rate,
            last_open_attempt: now - TimeDelta::milliseconds(RETRY_PERIOD_MS),
            last_active: now,
            next_sample_rate_to_probe: 0,
        })
    }

    fn try_reopen(&mut self) {
        let now = Time::now();
        if now - self.last_open_attempt >= TimeDelta::milliseconds(RETRY_PERIOD_MS) {
            self.last_open_attempt = now;
            let mut spec = self.spec.clone();
            if spec.sample_rate.is_none() {
                spec.sample_rate = Some(TARGET_SAMPLE_RATES[self.next_sample_rate_to_probe]);
                self.next_sample_rate_to_probe =
                    (self.next_sample_rate_to_probe + 1) % TARGET_SAMPLE_RATES.len();
            }
            match AlsaInput::open(spec, self.period_duration) {
                Ok(input) => {
                    self.input = Some(input);
                    self.last_active = now;
                }
                Err(_) => (),
            };
        }
    }
}

impl Input for ResilientAlsaInput {
    fn read(&mut self) -> Result<Option<Frame>> {
        if self.input.is_none() {
            self.try_reopen()
        }
        let mut reset = false;
        let result = match self.input.as_mut() {
            Some(input) => match input.read() {
                Ok(Some(frame)) => {
                    self.last_active = Time::now();
                    Some(frame)
                }
                Ok(None) => {
                    if self.probe_sample_rate
                        && Time::now() - self.last_active > TimeDelta::milliseconds(PROBE_TIME_MS)
                    {
                        reset = true;
                    }
                    None
                }
                Err(e) => {
                    println!("WARNING: input read from {} failed: {}", &self.spec.name, e);
                    reset = true;
                    None
                }
            },
            None => None,
        };
        if reset {
            self.input = None;
        }
        if self.input.is_none() {
            std::thread::sleep(self.period_duration.as_duration());
        }

        Ok(result)
    }

    fn min_delay(&self) -> TimeDelta {
        match self.input.as_ref() {
            Some(input) => input.min_delay(),
            None => self.period_duration,
        }
    }
}
