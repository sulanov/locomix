extern crate alsa;
extern crate nix;

use base::*;
use std::collections::VecDeque;
use std::cmp;
use std::ffi::CString;
use std::sync::mpsc;
use std::thread;
use time::{Time, TimeDelta};
use resampler;

pub trait Output: Send {
    fn write(&mut self, frame: Frame) -> Result<()>;
    fn deactivate(&mut self);
    fn sample_rate(&self) -> usize;
    fn min_delay(&self) -> TimeDelta;
    fn measured_sample_rate(&self) -> f64;
}

const RATE_DETECTION_PERIOD_MS: i64 = 30000;
const RATE_DETECTION_MIN_PERIOD_MS: i64 = 10000;
const BUFFER_PERIODS: usize = 4;

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
        while self.history.len() > 0
            && (now - self.history[0].time).in_seconds() >= RATE_DETECTION_PERIOD_MS as i64
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
    period_size: usize,
    format: SampleFormat,
    channels: Vec<ChannelPos>,
    min_delay: TimeDelta,
    rate_detector: RateDetector,
    measured_sample_rate: f64,

    out_queue: VecDeque<Frame>,
    buffer: Vec<u8>,
    buffer_pos: usize,
}

impl AlsaOutput {
    pub fn open(spec: DeviceSpec, period_duration: TimeDelta) -> Result<AlsaOutput> {
        let pcm = try!(alsa::PCM::open(
            &*CString::new(&spec.id[..]).unwrap(),
            alsa::Direction::Playback,
            false
        ));

        let sample_rate;
        let format;
        let period_size;
        {
            let hwp = try!(alsa::pcm::HwParams::any(&pcm));
            try!(hwp.set_channels(spec.channels.len() as u32));

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

            sample_rate = match spec.sample_rate {
                Some(rate) => {
                    try!(hwp.set_rate(rate as u32, alsa::ValueOr::Nearest));
                    rate
                }
                None => {
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
                        return Err(Error::new(&format!(
                            "Can't find appropriate sample rate for {} ({})",
                            &spec.name, &spec.id
                        )));
                    }
                    selected
                }
            };

            let target_period_size = period_duration * sample_rate as i64 / TimeDelta::seconds(1);
            try!(hwp.set_period_size_near(
                target_period_size as alsa::pcm::Frames,
                alsa::ValueOr::Nearest
            ));
            try!(hwp.set_periods(BUFFER_PERIODS as u32, alsa::ValueOr::Nearest));
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
                "INFO: Opened {} ({}). Buffer size: {}x{}. {} {} {}ch. Chmap: {}",
                spec.name,
                spec.id,
                period_size,
                try!(hwp.get_periods()),
                sample_rate,
                format,
                spec.channels.len(),
                try!(pcm.get_chmap())
            );
        }

        let (_, device_delay) = try!(pcm.avail_delay());
        let min_delay = samples_to_timedelta(sample_rate, device_delay)
            + period_duration * (BUFFER_PERIODS as i64 + 2);

        Ok(AlsaOutput {
            pcm: pcm,
            sample_rate: sample_rate,
            channels: spec.channels,
            period_size: period_size,
            format: format,
            min_delay: min_delay,
            rate_detector: RateDetector::new(sample_rate as f64),
            measured_sample_rate: sample_rate as f64,
            out_queue: VecDeque::new(),
            buffer: vec![],
            buffer_pos: 0,
        })
    }

    fn next_buffer(&mut self, time: Time) {
        let bytes_per_frame = self.channels.len() * self.format.bytes_per_sample();
        self.buffer_pos = 0;

        loop {
            let frame = self.out_queue.pop_front();
            if frame.is_none() {
                println!("Empty {}", self.period_size);
                self.buffer = vec![0u8; self.period_size as usize * bytes_per_frame];
                break;
            }

            let frame = frame.unwrap();
            if frame.timestamp - time > TimeDelta::milliseconds(1) {
                let gap_samples = ((frame.timestamp - time) * self.sample_rate as i64
                    / TimeDelta::seconds(1)) as usize;
                let samples_to_fill = cmp::min(self.period_size, gap_samples);
                self.buffer = vec![0u8; samples_to_fill * bytes_per_frame];

                // Push the frame back to queue, it's not useful yet.
                self.out_queue.push_front(frame);
            } else if time - frame.timestamp > TimeDelta::milliseconds(1) {
                let samples_to_skip = ((time - frame.timestamp) * self.sample_rate as i64
                    / TimeDelta::seconds(1)) as usize;
                if samples_to_skip >= frame.len() {
                    // Drop this frame - it's too old
                    continue;
                } else {
                    self.buffer =
                        frame.to_buffer_with_channel_map(self.format, self.channels.as_slice());
                    self.buffer.drain(..samples_to_skip * bytes_per_frame);
                }
            } else {
                self.buffer =
                    frame.to_buffer_with_channel_map(self.format, self.channels.as_slice());
            }
            break;
        }
    }

    fn write_loop(&mut self) -> alsa::Result<()> {
        loop {
            let (avail, delay) = try!(self.pcm.avail_delay());
            if avail == 0 {
                break;
            }

            if self.buffer_pos >= self.buffer.len() {
                let time = Time::now() + samples_to_timedelta(self.sample_rate, delay);
                self.next_buffer(time)
            }

            let bytes_per_frame = self.channels.len() * self.format.bytes_per_sample();
            let bytes_to_write = cmp::min(
                avail as usize * bytes_per_frame,
                self.buffer.len() - self.buffer_pos,
            );
            let end = self.buffer_pos + bytes_to_write;
            let frames_written =
                try!(self.pcm.io().writei(&self.buffer[self.buffer_pos..end])) as usize;
            self.buffer_pos += frames_written * bytes_per_frame;

            match self.rate_detector
                .update(frames_written, avail as usize - frames_written)
            {
                None => (),
                Some(rate) => self.measured_sample_rate = rate,
            }
        }
        Ok(())
    }
}

impl Output for AlsaOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        if self.sample_rate != frame.sample_rate {
            println!(
                "WARNING: different sample rate, expected {}, received {}",
                self.sample_rate, frame.sample_rate
            );
            return Ok(());
        }

        self.out_queue.push_back(frame);

        match self.write_loop() {
            Ok(_) => (),
            Err(e) => {
                println!("Recovering output {}", e);
                self.rate_detector.reset();
                try!(
                    self.pcm
                        .recover(e.errno().unwrap_or(nix::Errno::UnknownErrno) as i32, true)
                );
            }
        }

        Ok(())
    }

    fn deactivate(&mut self) {}

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn min_delay(&self) -> TimeDelta {
        self.min_delay
    }

    fn measured_sample_rate(&self) -> f64 {
        self.measured_sample_rate
    }
}

const RETRY_PERIOD_SECS: i64 = 3;

pub struct ResilientAlsaOutput {
    spec: DeviceSpec,
    output: Option<AlsaOutput>,
    period_duration: TimeDelta,
    last_open_attempt: Time,
    sample_rate: usize,
    min_delay: TimeDelta,
}

impl ResilientAlsaOutput {
    pub fn new(spec: DeviceSpec, period_duration: TimeDelta) -> Box<Output> {
        let sample_rate;
        let device = match AlsaOutput::open(spec.clone(), period_duration) {
            Ok(device) => {
                sample_rate = device.sample_rate();
                Some(device)
            }
            Err(e) => {
                sample_rate = spec.sample_rate.unwrap_or(48000);
                println!("WARNING: failed to open {} ({}): {}", spec.name, spec.id, e);
                None
            }
        };
        Box::new(ResilientAlsaOutput {
            spec: spec,
            output: device,
            period_duration: period_duration,
            last_open_attempt: Time::now(),
            sample_rate: sample_rate,
            min_delay: period_duration * BUFFER_PERIODS as i64 + TimeDelta::milliseconds(20),
        })
    }

    fn try_reopen(&mut self) {
        let now = Time::now();
        if now - self.last_open_attempt >= TimeDelta::seconds(RETRY_PERIOD_SECS) {
            self.last_open_attempt = now;
            match AlsaOutput::open(self.spec.clone(), self.period_duration) {
                Ok(output) => {
                    self.sample_rate = output.sample_rate();
                    self.min_delay = output.min_delay();
                    self.output = Some(output);
                }
                Err(_) => println!(
                    "INFO: Failed to reopen {} ({})",
                    &self.spec.name, self.spec.id
                ),
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
                Err(_) => {
                    println!(
                        "WARNING: Write to {} ({}) failed",
                        &self.spec.name, &self.spec.id
                    );
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
    fn min_delay(&self) -> TimeDelta {
        self.min_delay
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
    delay: TimeDelta,
}

impl ResamplingOutput {
    pub fn new(output: Box<Output>, window_size: usize) -> Box<Output> {
        let resampler = resampler::StreamResampler::new(output.sample_rate(), window_size);
        let delay = samples_to_timedelta(output.sample_rate(), window_size as i64);
        Box::new(ResamplingOutput {
            output: output,
            resampler: resampler,
            delay: delay,
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

        match self.resampler.resample(frame) {
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

    fn min_delay(&self) -> TimeDelta {
        self.output.min_delay() + self.delay
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
    delay: TimeDelta,
}

impl FineResamplingOutput {
    pub fn new(output: Box<Output>, window_size: usize) -> Box<Output> {
        let resampler = resampler::FineStreamResampler::new(
            output.measured_sample_rate(),
            output.sample_rate(),
            window_size,
        );
        let delay = samples_to_timedelta(output.sample_rate(), window_size as i64);
        Box::new(FineResamplingOutput {
            output: output,
            resampler: resampler,
            last_rate_update: Time::now(),
            delay: delay,
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
        if (current_rate - new_rate).abs() / current_rate > 0.05
            || now - self.last_rate_update >= TimeDelta::milliseconds(RATE_UPDATE_PERIOD_MS)
        {
            println!("Output rate: {}", new_rate);
            self.resampler
                .set_output_sample_rate(new_rate, self.output.sample_rate());
            self.last_rate_update = now;
        }

        match self.resampler.resample(frame) {
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

    fn min_delay(&self) -> TimeDelta {
        self.output.min_delay() + self.delay
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
    SampleRate(usize),
    MeasuredSampleRate(f64),
    MinDelay(TimeDelta),
}

pub struct AsyncOutput {
    sender: mpsc::Sender<PipeMessage>,
    feedback_receiver: mpsc::Receiver<FeedbackMessage>,
    sample_rate: usize,
    min_delay: TimeDelta,
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
            min_delay: output.min_delay(),
            measured_sample_rate: output.measured_sample_rate(),
        };

        thread::spawn(move || {
            let mut sample_rate = 0;
            let mut measured_sample_rate = 0f64;
            let mut min_delay = TimeDelta::zero();
            loop {
                if sample_rate != output.sample_rate() {
                    sample_rate = output.sample_rate();
                    feedback_sender
                        .send(FeedbackMessage::SampleRate(sample_rate))
                        .expect("Failed to send feedback message.");
                }

                let current_measured_sample_rate = output.measured_sample_rate();
                if current_measured_sample_rate != measured_sample_rate {
                    measured_sample_rate = current_measured_sample_rate;
                    feedback_sender
                        .send(FeedbackMessage::MeasuredSampleRate(measured_sample_rate))
                        .expect("Failed to send feedback message.");
                }

                if min_delay != output.min_delay() {
                    min_delay = output.min_delay();
                    feedback_sender
                        .send(FeedbackMessage::MinDelay(min_delay))
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

                // let deadline = Time::now() + output.min_delay();
                // if frame.timestamp < deadline {
                //     println!(
                //         "ERROR: Dropping frame: Missed target output time. {:?}",
                //         deadline - frame.timestamp
                //     );
                //     continue;
                // }

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
                FeedbackMessage::SampleRate(sample_rate) => {
                    self.sample_rate = sample_rate;
                }
                FeedbackMessage::MeasuredSampleRate(value) => {
                    self.measured_sample_rate = value;
                }
                FeedbackMessage::MinDelay(value) => {
                    self.min_delay = value;
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

    fn min_delay(&self) -> TimeDelta {
        self.min_delay
    }

    fn measured_sample_rate(&self) -> f64 {
        self.measured_sample_rate
    }
}
