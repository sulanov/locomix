extern crate alsa;

use base::*;
use std;
use std::collections::VecDeque;
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
    fn measured_sample_rate(&self) -> f64;
}

const RATE_DETECTION_PERIOD_MS: i64 = 30000;
const RATE_DETECTION_MIN_PERIOD_MS: i64 = 10000;

struct HistoryItem {
    time: Time,
    size: usize,
    avail: usize,
}

struct RateDetector {
    history: VecDeque<HistoryItem>,
    sum: usize,
    expected_rate: f64,
}

impl RateDetector {
    fn new(expected_rate: f64) -> RateDetector {
        return RateDetector {
            history: VecDeque::new(),
            sum: 0,
            expected_rate: expected_rate,
        };
    }

    fn reset(&mut self) {
        self.history.clear();
        self.sum = 0;
    }

    fn update(&mut self, samples: usize, avail: usize) -> Option<f64> {
        let now = Time::now();
        self.sum += samples;
        self.history.push_back(HistoryItem {
            time: now,
            size: samples,
            avail: avail,
        });
        while self.history.len() > 0 &&
            (now - self.history[0].time).in_seconds() >= RATE_DETECTION_PERIOD_MS as i64
        {
            self.sum -= self.history.pop_front().unwrap().size;
        }

        let period = now - self.history[0].time;
        if period < TimeDelta::milliseconds(RATE_DETECTION_MIN_PERIOD_MS) {
            return None;
        }

        let samples_played = self.sum - self.history[0].size + avail - self.history[0].avail;
        let current_rate = samples_played as f64 / period.in_seconds_f();

        // Allow 2% deviation from the target.
        if (current_rate - self.expected_rate).abs() / self.expected_rate < 0.02 {
            Some(current_rate)
        } else {
            None
        }
    }
}

pub struct AlsaOutput {
    pcm: alsa::PCM,
    sample_rate: usize,
    period_duration: TimeDelta,
    period_size: usize,
    format: SampleFormat,
    rate_detector: RateDetector,
    measured_sample_rate: f64,
}

impl AlsaOutput {
    pub fn open(
        name: &str,
        target_sample_rate: usize,
        period_duration: TimeDelta,
    ) -> Result<AlsaOutput> {
        let pcm = try!(alsa::PCM::open(
            &*CString::new(name).unwrap(),
            alsa::Direction::Playback,
            false
        ));

        let sample_rate;
        let period_size;
        let format;
        {
            let hwp = try!(alsa::pcm::HwParams::any(&pcm));
            try!(hwp.set_channels(2));

            try!(hwp.set_access(alsa::pcm::Access::RWInterleaved));
            try!(hwp.set_rate_resample(false));

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

            if target_sample_rate > 0 {
                try!(hwp.set_rate(
                    target_sample_rate as u32,
                    alsa::ValueOr::Nearest
                ));
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
                    return Err(Error::new(
                        &format!("Can't find appropriate sample rate for {}", name),
                    ));
                }
                sample_rate = selected;
            }

            let target_period_size = period_duration * sample_rate as i64 / TimeDelta::seconds(1);
            try!(hwp.set_period_size_near(
                target_period_size as alsa::pcm::Frames,
                alsa::ValueOr::Nearest
            ));
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

            println!(
                "INFO: Opened {}. Buffer size: {}x{}. {} {}.",
                name,
                period_size,
                try!(hwp.get_periods()),
                sample_rate,
                format
            );
        }

        Ok(AlsaOutput {
            pcm: pcm,
            sample_rate: sample_rate,
            period_duration: period_duration,
            period_size: period_size,
            format: format,
            rate_detector: RateDetector::new(sample_rate as f64),
            measured_sample_rate: sample_rate as f64,
        })
    }
}

impl Output for AlsaOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        if self.sample_rate != frame.sample_rate {
            println!(
                "WARNING: different sample rate, expected {}, received {}",
                self.sample_rate,
                frame.sample_rate
            );
            return Ok(());
        }

        // Sleep if the frame is too far into the future.
        let now = Time::now();
        if frame.timestamp - now > self.period_duration * 6 {
            std::thread::sleep(
                (frame.timestamp - now - self.period_duration * 2).as_duration(),
            );
        }

        let buf = frame.to_buffer(self.format);
        let mut pos: usize = 0;

        while pos < frame.len() {
            let start = pos * self.format.bytes_per_sample() * CHANNELS;
            let r = self.pcm.io().writei(&buf[start..]);
            match r {
                Ok(l) => pos += l,
                Err(e) => {
                    println!("Recovering output {}", e);
                    try!(self.pcm.recover(e.code(), true));

                    let zero_buf =
                        vec![0u8; self.period_size * CHANNELS * self.format.bytes_per_sample()];
                    try!(self.pcm.io().writei(&zero_buf[..]));
                    try!(self.pcm.io().writei(&zero_buf[..]));
                    try!(self.pcm.io().writei(&zero_buf[..]));

                    self.rate_detector.reset();
                }
            }
        }

        match self.rate_detector
            .update(frame.len(), try!(self.pcm.avail()) as usize)
        {
            None => (),
            Some(rate) => self.measured_sample_rate = rate,
        }

        Ok(())
    }

    fn deactivate(&mut self) {}
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }
    fn period_size(&self) -> usize {
        self.period_size
    }
    fn measured_sample_rate(&self) -> f64 {
        self.measured_sample_rate
    }
}

const RETRY_PERIOD_SECS: i64 = 3;

pub struct ResilientAlsaOutput {
    device_name: String,
    output: Option<AlsaOutput>,
    target_sample_rate: usize,
    period_duration: TimeDelta,
    last_open_attempt: Time,
    sample_rate: usize,
    period_size: usize,
}

impl ResilientAlsaOutput {
    pub fn new(name: &str, target_sample_rate: usize, period_duration: TimeDelta) -> Box<Output> {
        let sample_rate;
        let period_size;
        let device = match AlsaOutput::open(name, target_sample_rate, period_duration) {
            Ok(device) => {
                sample_rate = device.sample_rate();
                period_size = device.period_size();
                Some(device)
            }
            Err(e) => {
                sample_rate = 48000;
                period_size =
                    (period_duration * sample_rate as i64 / TimeDelta::seconds(1)) as usize;
                println!("WARNING: failed to open {}: {}", name, e);
                None
            }
        };
        Box::new(ResilientAlsaOutput {
            device_name: String::from(name),
            output: device,
            target_sample_rate: target_sample_rate,
            period_duration: period_duration,
            last_open_attempt: Time::now(),
            sample_rate: sample_rate,
            period_size: period_size,
        })
    }

    fn try_reopen(&mut self) {
        let now = Time::now();
        if now - self.last_open_attempt >= TimeDelta::seconds(RETRY_PERIOD_SECS) {
            self.last_open_attempt = now;
            match AlsaOutput::open(
                &self.device_name,
                self.target_sample_rate,
                self.period_duration,
            ) {
                Ok(output) => {
                    self.sample_rate = output.sample_rate();
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
                Ok(()) => {
                    let out_rate = self.output.as_mut().unwrap().sample_rate();
                    if out_rate != self.sample_rate {
                        self.sample_rate = out_rate;
                    }
                }
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
        self.sample_rate
    }
    fn period_size(&self) -> usize {
        self.period_size
    }
    fn measured_sample_rate(&self) -> f64 {
        match self.output.as_ref() {
            Some(ref o) => o.measured_sample_rate(),
            None => self.sample_rate as f64,
        }
    }
}

pub struct ResamplingOutput {
    output: Box<Output>,
    resampler: resampler::StreamResampler,
}

impl ResamplingOutput {
    pub fn new(output: Box<Output>) -> Box<Output> {
        let resampler = resampler::StreamResampler::new(output.sample_rate());
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
            Some(aframe) => self.output.write(aframe),
        }
    }

    fn deactivate(&mut self) {
        self.output.deactivate();
    }

    fn sample_rate(&self) -> usize {
        // FIXME
        self.output.sample_rate()
    }

    fn period_size(&self) -> usize {
        // FIXME
        256
    }

    fn measured_sample_rate(&self) -> f64 {
        // FIXME
        self.sample_rate() as f64
    }
}

const RATE_UPDATE_PERIOD_MS: i64 = 1000;

pub struct FineResamplingOutput {
    output: Box<Output>,
    resampler: resampler::FineStreamResampler,
    last_rate_update: Time,
}

impl FineResamplingOutput {
    pub fn new(output: Box<Output>) -> Box<Output> {
        let resampler = resampler::FineStreamResampler::new(
            output.measured_sample_rate(),
            output.sample_rate(),
        );
        Box::new(FineResamplingOutput {
            output: output,
            resampler: resampler,
            last_rate_update: Time::now(),
        })
    }
}

impl Output for FineResamplingOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        let out_sample_rate = self.output.sample_rate();
        if out_sample_rate == 0 {
            return self.output.write(frame);
        }

        let now = Time::now();
        let current_rate = self.resampler.get_output_sample_rate();
        let new_rate = self.output.measured_sample_rate();
        if (current_rate - new_rate).abs() / current_rate > 0.05 ||
            now - self.last_rate_update >= TimeDelta::milliseconds(RATE_UPDATE_PERIOD_MS)
        {
            println!("Output rate: {}", new_rate);
            self.resampler
                .set_output_sample_rate(new_rate, self.output.sample_rate());
            self.last_rate_update = now;
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
        // FIXME
        self.output.sample_rate()
    }

    fn period_size(&self) -> usize {
        // FIXME
        256
    }

    fn measured_sample_rate(&self) -> f64 {
        // FIXME
        self.sample_rate() as f64
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
    MeasuredSampleRate(f64),
}

pub struct AsyncOutput {
    sender: mpsc::Sender<PipeMessage>,
    feedback_receiver: mpsc::Receiver<FeedbackMessage>,
    sample_rate: usize,
    period_size: usize,
    measured_sample_rate: f64,
}

impl AsyncOutput {
    pub fn new(mut output: Box<Output>) -> Box<Output> {
        let (sender, receiver) = mpsc::channel();
        let (feedback_sender, feedback_receiver) = mpsc::channel();

        let result = AsyncOutput {
            sender: sender,
            feedback_receiver: feedback_receiver,
            sample_rate: output.sample_rate(),
            period_size: output.period_size(),
            measured_sample_rate: output.measured_sample_rate(),
        };

        thread::spawn(move || {
            let mut sample_rate = 0;
            let mut period_size = 0;
            let mut measured_sample_rate = 0f64;
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

                let current_measured_sample_rate = output.measured_sample_rate();
                if current_measured_sample_rate != measured_sample_rate {
                    measured_sample_rate = current_measured_sample_rate;
                    feedback_sender
                        .send(FeedbackMessage::MeasuredSampleRate(measured_sample_rate))
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
                    println!(
                        "ERROR: Dropping frame: Missed target output time. {:?}",
                        now - frame.timestamp
                    );
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
                FeedbackMessage::MeasuredSampleRate(value) => {
                    self.measured_sample_rate = value;
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

    fn measured_sample_rate(&self) -> f64 {
        self.measured_sample_rate
    }
}
