extern crate alsa;

use std;
use std::error;
use std::time::{Instant, Duration};
use std::ffi::CString;
use base::*;

pub trait Output {
    fn write(&mut self, frame: &Frame) -> Result<()>;
    fn deactivate(&mut self);
    fn sample_rate(&self) -> usize;
    fn period_size(&self) -> usize;
}

pub struct AlsaOutput {
    pcm: alsa::PCM,
    sample_rate: usize,
    format: SampleFormat,
    period_size: usize,
}

impl AlsaOutput {
    pub fn open(name: &str, sample_rate: usize) -> Result<AlsaOutput> {
        let pcm = try!(alsa::PCM::open(&*CString::new(name).unwrap(),
                                       alsa::Direction::Playback,
                                       false));

        let period_size;
        let format;
        {
            let hwp = try!(alsa::pcm::HwParams::any(&pcm));
            try!(hwp.set_channels(2));
            try!(hwp.set_rate(sample_rate as u32, alsa::ValueOr::Nearest));

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

            try!(hwp.set_period_size_near((sample_rate * FRAME_SIZE_APPROX_MS / 1000) as
                                          alsa::pcm::Frames,
                                          alsa::ValueOr::Nearest));
            try!(hwp.set_periods(3, alsa::ValueOr::Nearest));
            try!(hwp.set_access(alsa::pcm::Access::RWInterleaved));
            try!(hwp.set_rate_resample(false));
            try!(pcm.hw_params(&hwp));

            period_size = try!(hwp.get_period_size()) as usize;
            format = match try!(hwp.get_format()) {
                alsa::pcm::Format::S16LE => SampleFormat::S16LE,
                alsa::pcm::Format::S243LE => SampleFormat::S24LE3,
                alsa::pcm::Format::S24LE => SampleFormat::S24LE4,
                alsa::pcm::Format::S32LE => SampleFormat::S32LE,
                fmt => return Err(Error::new(&format!("Unknown format: {:?}", fmt))),
            };

            println!("INFO: Opened {}. Buffer size: {}x{}. {} {}.",
                     name,
                     period_size,
                     try!(hwp.get_periods()),
                     sample_rate,
                     format);
        }

        Ok(AlsaOutput {
               pcm: pcm,
               sample_rate: sample_rate,
               format: format,
               period_size: period_size,
           })
    }
}

impl Output for AlsaOutput {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        let buf = frame.to_buffer(self.format);
        let r = self.pcm.io().writei(&buf[..]);
        match r {
            Ok(l) => {
                if l != frame.len() {
                    Err(Error::new(&format!("Partial write: {}/{}", l, frame.len())))
                } else {
                    Ok(())
                }
            }
            Err(e) => {
                println!("Recovering output {}", e);
                match self.pcm.recover(e.code(), true) {
                    Ok(_) => self.write(frame),
                    Err(_) => Err(Error::new(error::Error::description(&e))),
                }
            }
        }
    }

    fn deactivate(&mut self) {}

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn period_size(&self) -> usize {
        self.period_size
    }
}

const RETRY_PERIOD_SECS: u64 = 3;

pub struct ResilientAlsaOutput {
    device_name: String,
    output: Option<AlsaOutput>,
    sample_rate: usize,
    last_open_attempt: Instant,
    period_size: usize,
}

impl ResilientAlsaOutput {
    pub fn new(name: &str, sample_rate: usize) -> ResilientAlsaOutput {
        let mut period_size = sample_rate * FRAME_SIZE_APPROX_MS / 1000;
        let device = match AlsaOutput::open(name, sample_rate) {
            Ok(device) => {
                period_size = device.period_size();
                Some(device)
            }
            Err(e) => {
                println!("WARNING: failed to open {}: {}", name, e);
                None
            }
        };
        ResilientAlsaOutput {
            device_name: String::from(name),
            output: device,
            sample_rate: sample_rate,
            last_open_attempt: Instant::now(),
            period_size: period_size,
        }
    }

    fn try_reopen(&mut self) {
        let now = Instant::now();
        if (now - self.last_open_attempt).as_secs() >= RETRY_PERIOD_SECS {
            self.last_open_attempt = now;
            match AlsaOutput::open(&self.device_name, self.sample_rate) {
                Ok(output) => {
                    self.period_size = output.period_size();
                    self.output = Some(output);
                }
                Err(e) => println!("INFO: Failed to reopen {}", e),
            }
        }
    }
}

impl Output for ResilientAlsaOutput {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        if self.output.is_none() {
            self.try_reopen()
        }

        if self.output.is_some() {
            let mut reset = false;
            match self.output.as_mut().unwrap().write(frame) {
                Ok(()) => (),
                Err(e) => {
                    println!("warning: write to {} failed: {}", self.device_name, e);
                    reset = true;
                }
            }
            if reset {
                self.output = None;
            }
        }

        if self.output.is_none() {
            std::thread::sleep(Duration::from_millis((frame.len() * 1000 / self.sample_rate) as
                                                     u64));
        }

        Ok(())
    }

    fn deactivate(&mut self) {
        self.output = None;
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn period_size(&self) -> usize {
        self.period_size
    }
}
