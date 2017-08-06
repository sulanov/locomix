extern crate alsa;

use base::*;
use std;
use std::ffi::CString;
use std::sync::mpsc;
use std::thread;
use time::{Time, TimeDelta};
use resampler;

pub trait Output: Send {
    fn write(&mut self, frame: Frame) -> Result<()>;
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
    pub fn open(name: &str, target_sample_rate: usize) -> Result<AlsaOutput> {
        let pcm = try!(alsa::PCM::open(&*CString::new(name).unwrap(),
                                       alsa::Direction::Playback,
                                       false));

        let sample_rate;
        let period_size;
        let format;
        {
            let hwp = try!(alsa::pcm::HwParams::any(&pcm));
            try!(hwp.set_channels(2));

            try!(hwp.set_access(alsa::pcm::Access::RWInterleaved));
            try!(hwp.set_rate_resample(false));

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

            if target_sample_rate > 0 {
                try!(hwp.set_rate(target_sample_rate as u32, alsa::ValueOr::Nearest));
                sample_rate = target_sample_rate;
            } else {
                let mut selected: usize = 0;
                for rate in [192000, 96000, 88200, 48000, 44100].iter() {
                    match hwp.set_rate(*rate as u32, alsa::ValueOr::Nearest) {
                        Ok(_) => {
                            selected = *rate;
                            break;
                        }
                        Err(_) => (),
                    }
                }
                if selected == 0 {
                    return Err(Error::new(&format!("Can't find appropriate sample rate for {}",
                                                   name)));
                }
                sample_rate = selected;
            }

            try!(hwp.set_period_size_near((sample_rate * FRAME_SIZE_MS / 1000) as
                                          alsa::pcm::Frames,
                                          alsa::ValueOr::Nearest));
            try!(hwp.set_periods(3, alsa::ValueOr::Nearest));
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
    fn write(&mut self, frame: Frame) -> Result<()> {
        if self.sample_rate != frame.sample_rate {
            println!("WARNING: different sample rate, expected {}, received {}",
                     self.sample_rate,
                     frame.sample_rate);
            return Ok(());
        }

        // Sleep if the frame is too far into the future.
        let now = Time::now();
        if frame.timestamp - now > TimeDelta::milliseconds(FRAME_SIZE_MS as i64) * 5 {
            std::thread::sleep((frame.timestamp - now -
                                TimeDelta::milliseconds(FRAME_SIZE_MS as i64) * 1)
                                       .as_duration());
        }

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
                try!(self.pcm.recover(e.code(), true));
                self.write(frame)
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

const RETRY_PERIOD_SECS: i64 = 3;

pub struct ResilientAlsaOutput {
    device_name: String,
    output: Option<AlsaOutput>,
    target_sample_rate: usize,
    last_open_attempt: Time,
    period_size: usize,
}

impl ResilientAlsaOutput {
    pub fn new(name: &str, target_sample_rate: usize) -> Box<Output> {
        let mut period_size = target_sample_rate * FRAME_SIZE_MS / 1000;
        let device = match AlsaOutput::open(name, target_sample_rate) {
            Ok(device) => {
                period_size = device.period_size();
                Some(device)
            }
            Err(e) => {
                println!("WARNING: failed to open {}: {}", name, e);
                None
            }
        };
        Box::new(ResilientAlsaOutput {
                     device_name: String::from(name),
                     output: device,
                     target_sample_rate: target_sample_rate,
                     last_open_attempt: Time::now(),
                     period_size: period_size,
                 })
    }

    fn try_reopen(&mut self) {
        let now = Time::now();
        if now - self.last_open_attempt >= TimeDelta::seconds(RETRY_PERIOD_SECS) {
            self.last_open_attempt = now;
            match AlsaOutput::open(&self.device_name, self.target_sample_rate) {
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
    fn write(&mut self, frame: Frame) -> Result<()> {
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

        Ok(())
    }

    fn deactivate(&mut self) {
        self.output = None;
    }

    fn sample_rate(&self) -> usize {
        match self.output.as_ref() {
            Some(o) => o.sample_rate(),
            None => 0,
        }
    }

    fn period_size(&self) -> usize {
        self.period_size
    }
}

pub struct ResamplingOutput {
    output: Box<Output>,
    resampler: resampler::StreamResampler,
}

impl ResamplingOutput {
    pub fn new(output: Box<Output>) -> Box<Output> {
        let mut resampler = resampler::StreamResampler::new(output.sample_rate());
        Box::new(ResamplingOutput {
                     output: output,
                     resampler: resampler,
                 })
    }
}

impl Output for ResamplingOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        let out_sample_rate = self.output.sample_rate();
        if out_sample_rate == 0 {
            return self.output.write(frame);
        }

        if self.resampler.get_output_sample_rate() != out_sample_rate {
            self.resampler.set_output_sample_rate(out_sample_rate);
        }

        match self.resampler.resample(&frame) {
            None => Ok(()),
            Some(frame) => self.output.write(frame),
        }
    }

    fn deactivate(&mut self) {
        self.output.deactivate();
    }

    fn sample_rate(&self) -> usize {
        self.output.sample_rate()
    }

    fn period_size(&self) -> usize {
        // FIXME
        256
    }
}


enum PipeMessage {
    Frame(Frame),
    Deactivate,
}

enum FeedbackMessage {
    NewConfig {
        sample_rate: usize,
        period_size: usize,
    },
}

pub struct AsyncOutput {
    sender: mpsc::Sender<PipeMessage>,
    feedback_receiver: mpsc::Receiver<FeedbackMessage>,
    sample_rate: usize,
    period_size: usize,
}

impl AsyncOutput {
    pub fn open(name: &str) -> Box<Output> {
        AsyncOutput::new(ResamplingOutput::new(AsyncOutput::new(ResilientAlsaOutput::new(name, 0))))
    }

    pub fn new(mut output: Box<Output>) -> Box<Output> {
        let (sender, receiver) = mpsc::channel();
        let (feedback_sender, feedback_receiver) = mpsc::channel();

        let result = AsyncOutput {
            sender: sender,
            feedback_receiver: feedback_receiver,
            sample_rate: output.sample_rate(),
            period_size: output.period_size() * FRAME_SIZE_MS / 1000,
        };

        thread::spawn(move || {
            let mut sample_rate = 0;
            let mut period_size = 0;
            loop {
                if sample_rate != output.sample_rate() || period_size != output.period_size() {
                    sample_rate = output.sample_rate();
                    period_size = output.period_size();
                    feedback_sender
                        .send(FeedbackMessage::NewConfig {
                                  sample_rate,
                                  period_size,
                              })
                        .expect("Failed to send feedback message.");
                }

                let frame = match receiver.recv() {
                    Ok(PipeMessage::Frame(frame)) => frame,
                    Ok(PipeMessage::Deactivate) => {
                        output.deactivate();
                        continue;
                    }
                    Err(_) => {
                        println!("ERROR: Failed to receive next output frame.");
                        return;
                    }
                };

                let now = Time::now();
                if frame.timestamp < now {
                    println!("ERROR: Dropping frame: Missed target output time. {:?}",
                             now - frame.timestamp);
                    continue;
                }

                output.write(frame).expect("Failed to write samples");
            }
        });

        Box::new(result)
    }
}

impl Output for AsyncOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        for msg in self.feedback_receiver.try_iter() {
            match msg {
                FeedbackMessage::NewConfig {
                    sample_rate,
                    period_size,
                } => {
                    self.sample_rate = sample_rate;
                    self.period_size = period_size;
                }
            }
        }

        self.sender
            .send(PipeMessage::Frame(frame))
            .expect("Failed to send frame");
        Ok(())
    }

    fn deactivate(&mut self) {
        self.sender
            .send(PipeMessage::Deactivate)
            .expect("Failed to deactivate");
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn period_size(&self) -> usize {
        self.period_size
    }
}
