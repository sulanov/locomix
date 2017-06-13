extern crate alsa;
extern crate libc;

use std;
use std::ffi::CString;
use std::error;
use std::time::{Instant, Duration};
use std::collections::VecDeque;
use base::*;

use super::input::*;

const ACCEPTED_RATES: [usize; 6] = [44100, 48000, 88200, 96000, 176400, 192000];

const RATE_DETECTION_SECONDS: usize = 1;
const RATE_DETECTION_MIN_PERIOD_MS: usize = 950;

struct RateDetector {
    history: VecDeque<(Instant, usize)>,
    sum: usize,
}

impl RateDetector {
    fn new() -> RateDetector {
        return RateDetector {
                   history: VecDeque::new(),
                   sum: 0,
               };
    }

    fn update(&mut self, samples: usize) -> Option<usize> {
        let now = Instant::now();
        self.sum += samples;
        self.history.push_back((now, samples));
        while self.history.len() > 0 &&
              (now - self.history[0].0).as_secs() >= RATE_DETECTION_SECONDS as u64 {
            self.sum -= self.history.pop_front().unwrap().1;
        }

        let period_ms = ((now - self.history[0].0).subsec_nanos() / 1000000) as usize;
        if period_ms < RATE_DETECTION_MIN_PERIOD_MS {
            return None;
        }

        let current_rate = (self.sum - self.history[0].1) * 1000 / period_ms;

        for rate in ACCEPTED_RATES.iter() {
            // Check if the current rate is within 5% of a known value
            if ((*rate as i32 - current_rate as i32).abs() as usize) < *rate / 20 {
                return Some(*rate);
            }
        }
        None
    }
}

enum State {
    Inactive,
    Active { silent_samples: usize },
}

pub struct AlsaInput {
    pcm: alsa::PCM,
    sample_rate: usize,
    format: SampleFormat,
    period_size: usize,
    rate_detector: RateDetector,
    state: State,
}

const TARGET_RATE: usize = 48000;
const INPUT_FRAME_SIZE: usize = 256;

// Deactivate input after 60 seconds of silence.
const SILENCE_PERIOD_SECONDS: usize = 60;

impl AlsaInput {
    pub fn open(name: &str) -> Result<AlsaInput> {
        let pcm = try!(alsa::PCM::open(&*CString::new(name).unwrap(),
                                       alsa::Direction::Capture,
                                       false));

        let sample_rate;
        let period_size;
        let format;
        {
            let hwp = try!(alsa::pcm::HwParams::any(&pcm));
            try!(hwp.set_channels(2));

            for fmt in [alsa::pcm::Format::S32LE,
                        alsa::pcm::Format::S243LE,
                        alsa::pcm::Format::S24LE,
                        alsa::pcm::Format::S16LE]
                        .iter() {
                match hwp.set_format(*fmt) {
                    Ok(_) => break,
                    Err(_) => (),
                }
            }

            try!(hwp.set_period_size_near(INPUT_FRAME_SIZE as alsa::pcm::Frames,
                                          alsa::ValueOr::Nearest));
            try!(hwp.set_periods(10, alsa::ValueOr::Nearest));
            try!(hwp.set_rate_resample(false));
            try!(hwp.set_rate(TARGET_RATE as u32, alsa::ValueOr::Nearest));
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

            println!("INFO: Reading {}. Buffer size: {}x{}. {} {}",
                     name,
                     period_size,
                     try!(hwp.get_periods()),
                     sample_rate,
                     format);
        }

        Ok(AlsaInput {
               pcm: pcm,
               sample_rate: sample_rate,
               format: format,
               period_size: period_size,
               rate_detector: RateDetector::new(),
               state: State::Inactive,
           })
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
                if e.code() == -libc::EWOULDBLOCK {
                    return Ok(None);
                }
                println!("Recovering AlsaInput {}", e.code());
                match self.pcm.recover(e.code(), true) {
                    Ok(_) => return self.read(),
                    Err(_) => return Err(Error::new(error::Error::description(&e))),
                }
            }
        };

        assert!(samples > 0);

        match self.rate_detector.update(samples) {
            Some(rate) => {
                if rate != self.sample_rate {
                    println!("INFO: rate changed {}", rate);
                    self.sample_rate = rate;
                }
            }
            None => {
                // Drop all data if we don't know sample rate.
                return Ok(None);
            }
        }

        let mut all_zeros = true;
        for i in 0..(samples * 2) {
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
                    State::Active { silent_samples: silence }
                }
            }
            (&State::Inactive, true) => State::Inactive,
        };

        match self.state {
            State::Inactive => return Ok(None),
            _ => (),
        }


        let bytes = samples * CHANNELS * self.format.bytes_per_sample();
        Ok(Some(Frame::from_buffer(self.format, self.sample_rate, &buf[0..bytes])))
    }

    fn samples_buffered(&mut self) -> Result<usize> {
        Ok(try!(self.pcm.status()).get_avail() as usize)
    }

    fn is_synchronized(&self) -> bool {
        true
    }
}

const RETRY_PERIOD_SECS: u64 = 3;

pub struct ResilientAlsaInput {
    device_name: String,
    input: Option<AlsaInput>,
    last_open_attempt: Instant,
}

impl ResilientAlsaInput {
    pub fn new(name: &str) -> ResilientAlsaInput {
        let device = match AlsaInput::open(name) {
            Ok(device) => Some(device),
            Err(e) => {
                println!("warning: failed to open {}: {}", name, e);
                None
            }
        };
        ResilientAlsaInput {
            device_name: String::from(name),
            input: device,
            last_open_attempt: Instant::now(),
        }
    }

    fn try_reopen(&mut self) {
        let now = Instant::now();
        if (now - self.last_open_attempt).as_secs() >= RETRY_PERIOD_SECS {
            self.last_open_attempt = now;
            match AlsaInput::open(&self.device_name) {
                Ok(input) => self.input = Some(input),
                Err(e) => println!("info: Failed to reopen {}", e),
            }
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
            Some(input) => {
                match input.read() {
                    Ok(Some(frame)) => Some(frame),
                    Ok(None) => None,
                    Err(e) => {
                        println!("warning: input read from {} failed: {}",
                                 self.device_name,
                                 e);
                        reset = true;
                        None
                    }
                }
            }
            None => None,
        };
        if reset {
            self.input = None;
        }
        if self.input.is_none() {
            std::thread::sleep(Duration::from_millis(FRAME_SIZE_APPROX_MS as u64));
        }

        Ok(result)
    }

    fn samples_buffered(&mut self) -> Result<usize> {
        let mut reset = false;
        let result = match self.input.as_mut() {
            Some(input) => {
                match input.samples_buffered() {
                    Ok(r) => Ok(r),
                    Err(e) => {
                        println!("warning: samples_buffered() call failed for {}: {}",
                                 self.device_name,
                                 e);
                        reset = true;
                        Ok(0)
                    }
                }
            }
            None => Ok(0),
        };

        if reset {
            self.input = None;
        }

        result
    }

    fn is_synchronized(&self) -> bool {
        true
    }
}
