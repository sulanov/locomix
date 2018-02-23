extern crate alsa;
extern crate libc;
extern crate nix;

use std;
use std::ffi::CString;
use std::error;
use time::{Time, TimeDelta};
use base::*;

use super::input::*;

const CHANNELS: usize = 2;

const ACCEPTED_RATES: [usize; 4] = [44100, 48000, 88200, 96000];
const MAX_SAMPLE_RATE_ERROR: f32 = 1000.0;

enum State {
    Inactive,
    Active { silent_samples: usize },
}

pub struct AlsaInput {
    spec: DeviceSpec,
    pcm: alsa::PCM,
    sample_rate: usize,
    format: SampleFormat,
    period_size: usize,
    rate_detector: RateDetector,
    exact_rate_detector: RateDetector,
    state: State,
    current_rate: DetectedRate,
    last_rate_update: Time,
}

// Deactivate input after 5 seconds of silence.
const SILENCE_PERIOD_SECONDS: usize = 5;

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

        Ok(Box::new(AlsaInput {
            spec: spec,
            pcm: pcm,
            sample_rate: sample_rate,
            format: format,
            period_size: period_size,
            rate_detector: RateDetector::new(MAX_SAMPLE_RATE_ERROR),
            exact_rate_detector: RateDetector::new(1.0),
            state: State::Inactive,
            current_rate: DetectedRate { rate: sample_rate as f32, error: sample_rate as f32 },
            last_rate_update: Time::now(),
        }))
    }
}

impl Input for AlsaInput {
    fn read(&mut self) -> Result<Option<Frame>> {
        let mut buf = vec![0u8; self.period_size * CHANNELS * self.format.bytes_per_sample()];
        let read_result = self.pcm.io().readi(&mut buf[..]);
        let samples = match read_result {
            Ok(0) => return Ok(None),
            Ok(r) => r,
            Err(e) => {
                if e.errno().unwrap_or(nix::Errno::UnknownErrno) == nix::errno::EWOULDBLOCK {
                    return Ok(None);
                }
                println!("Recovering AlsaInput {}: {:?}", &self.spec.name, e.errno());
                match self.pcm
                    .recover(e.errno().unwrap_or(nix::Errno::UnknownErrno) as i32, true)
                {
                    Ok(_) => return self.read(),
                    Err(_) => return Err(Error::new(error::Error::description(&e))),
                }
            }
        };

        assert!(samples > 0);

        let now = Time::now();
        let rate = self.rate_detector.update(samples, now);
        if (self.sample_rate as f32 - rate.rate).abs() > MAX_SAMPLE_RATE_ERROR &&
           rate.error < MAX_SAMPLE_RATE_ERROR {
            let mut new_rate = 0;
            for r in &ACCEPTED_RATES[..] {
                if (rate.rate - *r as f32).abs() < MAX_SAMPLE_RATE_ERROR {
                    new_rate = *r;
                }
            }

            if new_rate > 0 {
                println!(
                    "INFO: rate changed for input device {}: {}",
                    &self.spec.name, new_rate
                );
                self.sample_rate = new_rate;
                self.current_rate = DetectedRate { rate: new_rate as f32, error: MAX_SAMPLE_RATE_ERROR };
                self.exact_rate_detector.reset();
            } else if rate.error < 100.0 {
                println!(
                    "WARNING: Unknown sample rate for {}: {}",
                    &self.spec.name, rate.rate
                );
            }
        }

        if self.spec.exact_sample_rate {
            let rate = self.rate_detector.update(samples, now);

            if (now - self.last_rate_update) > TimeDelta::milliseconds(500) {
               self.last_rate_update = now;
               if rate.error < 10.0 {
                    self.current_rate = rate;
               }
            }
        }


        let bytes = samples * CHANNELS * self.format.bytes_per_sample();

        let mut all_zeros = true;
        for i in 0..bytes {
            if buf[i] != 0 {
                all_zeros = false;
                break;
            }
        }

        self.state = match (&self.state, all_zeros) {
            (_, false) => State::Active { silent_samples: 0 },
            (&State::Active { silent_samples }, true) => {
                let silence = silent_samples + samples;
                if silence > SILENCE_PERIOD_SECONDS * self.sample_rate {
                    State::Inactive
                } else {
                    State::Active {
                        silent_samples: silence,
                    }
                }
            }
            (&State::Inactive, true) => State::Inactive,
        };

        match self.state {
            State::Inactive => return Ok(None),
            _ => (),
        }

        let mut frame = Frame::from_buffer_stereo(
            self.format, self.current_rate.rate, &buf[0..bytes], Time::now());
        frame.timestamp -= frame.duration();

        Ok(Some(frame))
    }

    fn min_delay(&self) -> TimeDelta {
        samples_to_timedelta(self.sample_rate as f32, self.period_size as i64)
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
    pub fn new(spec: DeviceSpec, period_duration: TimeDelta, probe_sample_rate: bool) -> Box<ResilientAlsaInput> {
        Box::new(ResilientAlsaInput {
            spec: spec,
            period_duration: period_duration,
            input: None,
            probe_sample_rate: probe_sample_rate,
            last_open_attempt: Time::now(),
            last_active: Time::now(),
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
                    if self.probe_sample_rate &&
                       Time::now() - self.last_active > TimeDelta::milliseconds(PROBE_TIME_MS) {
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
