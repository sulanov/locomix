extern crate alsa;

use base::*;
use std;
use std::error;
use std::ffi::CString;
use std::sync::mpsc;
use std::thread;
use std::time::{Instant, Duration};
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

            try!(hwp.set_period_size_near((sample_rate * FRAME_SIZE_APPROX_MS / 1000) as
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
        assert!(self.sample_rate == frame.sample_rate);
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
    target_sample_rate: usize,
    last_open_attempt: Instant,
    period_size: usize,
}

impl ResilientAlsaOutput {
    pub fn new(name: &str, target_sample_rate: usize) -> ResilientAlsaOutput {
        let mut period_size = target_sample_rate * FRAME_SIZE_APPROX_MS / 1000;
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
        ResilientAlsaOutput {
            device_name: String::from(name),
            output: device,
            target_sample_rate: target_sample_rate,
            last_open_attempt: Instant::now(),
            period_size: period_size,
        }
    }

    pub fn try_reopen(&mut self) {
        let now = Instant::now();
        if (now - self.last_open_attempt).as_secs() >= RETRY_PERIOD_SECS {
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

        let frame_duration = Duration::from_millis((frame.len() * 1000 / self.sample_rate()) as
                                                   u64);

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
        } else {
            std::thread::sleep(frame_duration);
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

enum PipeMessage {
    Frame(Frame),
    Deactivate,
}

pub struct ResamplingOutput {
    sender: mpsc::SyncSender<PipeMessage>,
    sample_rate: usize,
    period_size: usize,
}

impl ResamplingOutput {
    pub fn open(name: &str, input_sample_rate: usize) -> ResamplingOutput {
        ResamplingOutput::new(ResilientAlsaOutput::new(name, 0), input_sample_rate)
    }

    pub fn new(mut output: ResilientAlsaOutput, input_sample_rate: usize) -> ResamplingOutput {
        let (sender, receiver) = mpsc::sync_channel(2);

        let result = ResamplingOutput {
            sender: sender,
            sample_rate: input_sample_rate,
            period_size: input_sample_rate * FRAME_SIZE_APPROX_MS / 1000,
        };

        thread::spawn(move || {
            let mut resampler_factory = resampler::ResamplerFactory::new();
            let mut current_rate = 0;
            let mut resamplers: Option<[resampler::FastResampler; 2]> = None;

            loop {
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

                if output.sample_rate() == 0 {
                    output.try_reopen();
                }

                let out_sample_rate = output.sample_rate();

                if out_sample_rate == 0 {
                    continue;
                }

                if current_rate != out_sample_rate {
                    current_rate = out_sample_rate;
                    resamplers = if input_sample_rate != out_sample_rate {
                        Some([resampler_factory.create_resampler(input_sample_rate,
                                                                 out_sample_rate,
                                                                 200),
                              resampler_factory.create_resampler(input_sample_rate,
                                                                 out_sample_rate,
                                                                 200)])
                    } else {
                        None
                    };
                }

                let resampled_frame = match resamplers.as_mut() {
                    None => frame,
                    Some(resamplers) => {
                        Frame {
                            sample_rate: out_sample_rate,
                            left: resamplers[0].resample(&frame.left),
                            right: resamplers[1].resample(&frame.right),
                        }
                    }
                };

                if resampled_frame.len() > 0 {
                    output.write(resampled_frame);
                }
            }
        });

        result
    }
}

impl Output for ResamplingOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        self.sender.send(PipeMessage::Frame(frame));
        Ok(())
    }
    fn deactivate(&mut self) {
        self.sender.send(PipeMessage::Deactivate);
    }
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }
    fn period_size(&self) -> usize {
        self.period_size
    }
}
