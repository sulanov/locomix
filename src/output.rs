extern crate alsa;
extern crate libc;
extern crate nix;

use base::*;
use resampler;
use std::cmp;
use std::ffi::CString;
use std::sync::mpsc;
use std::thread;
use time::{Time, TimeDelta};

pub trait Output: Send {
    fn write(&mut self, frame: Frame) -> Result<()>;
    fn deactivate(&mut self);
    fn sample_rate(&self) -> f64;
    fn min_delay(&self) -> TimeDelta;
}

const MAX_TIMESTAMP_DEVIATION_US: i64 = 500;
const BUFFER_PERIODS: usize = 4;
const RATE_UPDATE_PERIOD_MS: i64 = 500;

enum AlsaWriteLoopFeedback {
    CurrentRate(f64),
}

struct InterleavedFrame {
    timestamp: Time,
    buf: Vec<u8>,
}

struct OutputPosTracker {
    base_time: Time,
    sample_rate: f64,
    samples_pos: f64,
    offset: SeriesStats,
}

impl OutputPosTracker {
    fn new(sample_rate: f64) -> OutputPosTracker {
        OutputPosTracker {
            base_time: Time::zero(),
            sample_rate: sample_rate,
            samples_pos: 0.0,
            offset: SeriesStats::new(100),
        }
    }

    fn pos_no_offset(&self) -> Time {
        assert!(self.base_time != Time::zero());
        self.base_time + TimeDelta::seconds_f(self.samples_pos / self.sample_rate)
    }

    fn pos(&self) -> Time {
        self.pos_no_offset() + TimeDelta::seconds_f(self.offset.average().unwrap_or(0.0))
    }

    fn pos_estimate(&mut self, time: Time) {
        if self.base_time == Time::zero() {
            self.base_time = time;
            self.samples_pos = 0.0;
        }
        let new_offset = (time - self.pos_no_offset()).in_seconds_f();
        self.offset.push(new_offset);
    }

    fn add_samples(&mut self, samples: usize) {
        self.samples_pos += samples as f64;
    }

    fn reset(&mut self, base_time: Time) {
        self.base_time = base_time;
        self.samples_pos = 0.0;
        self.offset.reset();
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        self.base_time = self.pos_no_offset();
        self.sample_rate = sample_rate;
        self.samples_pos = 0.0;
    }
}

struct AlsaWriteLoop {
    spec: DeviceSpec,
    pcm: alsa::PCM,
    sample_rate: f64,
    period_size: usize,
    bytes_per_frame: usize,
    pos_tracker: OutputPosTracker,
    rate_detector: RateDetector,

    frame_receiver: mpsc::Receiver<InterleavedFrame>,
    feedback_sender: mpsc::Sender<AlsaWriteLoopFeedback>,
    cur_frame: Option<InterleavedFrame>,
    buffer: Vec<u8>,
    buffer_pos: usize,
    last_rate_update: Time,
}

#[derive(PartialEq)]
enum LoopState {
    Running,
    Stop,
}

impl AlsaWriteLoop {
    fn next_buffer(&mut self, time: Time) -> LoopState {
        self.buffer_pos = 0;

        loop {
            if self.cur_frame.is_none() {
                self.cur_frame = match self.frame_receiver.try_recv() {
                    Ok(frame) => Some(frame),
                    Err(mpsc::TryRecvError::Empty) => None,
                    Err(mpsc::TryRecvError::Disconnected) => return LoopState::Stop,
                };
            }

            if self.cur_frame.is_none() {
                self.buffer = vec![0u8; self.period_size as usize * self.bytes_per_frame];
                break;
            }

            let max_deviation = TimeDelta::microseconds(MAX_TIMESTAMP_DEVIATION_US);
            let frame = self.cur_frame.take().unwrap();
            if frame.timestamp - time > max_deviation {
                let gap_samples =
                    ((frame.timestamp - time).in_seconds_f() * self.sample_rate) as usize;
                let samples_to_fill = cmp::min(self.period_size, gap_samples);
                self.buffer = vec![0u8; samples_to_fill * self.bytes_per_frame];

                // Keep the frame and use it next time next_buffer() is called.
                self.cur_frame = Some(frame);
            } else if time - frame.timestamp > max_deviation {
                let samples_to_skip =
                    ((time - frame.timestamp).in_seconds_f() * self.sample_rate) as usize;
                let frame_len = frame.buf.len() / self.bytes_per_frame;
                if samples_to_skip >= frame_len {
                    // Drop this frame - it's too old.
                    continue;
                } else {
                    self.buffer = frame.buf;
                    self.buffer_pos = samples_to_skip * self.bytes_per_frame;
                }
            } else {
                self.buffer = frame.buf;
            }
            break;
        }

        LoopState::Running
    }

    fn start_device(&mut self) -> alsa::Result<()> {
        let buf = vec![0u8; self.bytes_per_frame * self.period_size];
        for _ in 0..BUFFER_PERIODS {
            self.pcm.io().writei(&buf[..])?;
        }
        self.pcm.start()?;
        for _ in 0..BUFFER_PERIODS {
            self.pcm.io().writei(&buf[..])?;
        }

        self.rate_detector.reset();
        let stream_time_estimate = self.stream_time_estimate()?;
        self.pos_tracker.reset(stream_time_estimate);

        Ok(())
    }

    fn stream_time_estimate(&mut self) -> alsa::Result<Time> {
        let now1 = Time::now();
        let delay = self.pcm.delay()?;
        let now2 = Time::now();
        let now = now1 + (now2 - now1) / 2;
        Ok(now + samples_to_timedelta(self.sample_rate, delay as i64) + self.spec.delay)
    }

    fn do_write(&mut self) -> alsa::Result<LoopState> {
        if self.buffer_pos >= self.buffer.len() {
            let stream_pos = self.pos_tracker.pos();
            if self.next_buffer(stream_pos) == LoopState::Stop {
                return Ok(LoopState::Stop);
            }
        }

        let frames_written = try!(self.pcm.io().writei(&self.buffer[self.buffer_pos..])) as usize;
        self.buffer_pos += frames_written * self.bytes_per_frame;

        let stream_time_estimate = self.stream_time_estimate()?;
        self.pos_tracker.add_samples(frames_written);
        self.pos_tracker.pos_estimate(stream_time_estimate);

        if self.spec.exact_sample_rate {
            let current_rate = self.rate_detector
                .update(frames_written, stream_time_estimate);
            let now = Time::now();
            if now - self.last_rate_update > TimeDelta::milliseconds(RATE_UPDATE_PERIOD_MS) {
                self.last_rate_update = now;
                if current_rate.error < 0.01 * self.sample_rate {
                    self.sample_rate = current_rate.rate;
                    self.pos_tracker.set_sample_rate(current_rate.rate as f64);
                    self.feedback_sender
                        .send(AlsaWriteLoopFeedback::CurrentRate(current_rate.rate))
                        .expect("Failed to send rate update");
                } else {
                    println!("WARNING: Erroneous output rate {}.", current_rate.rate);
                }
            }
        }

        Ok(LoopState::Running)
    }

    fn run_loop_err(&mut self) -> alsa::Result<()> {
        self.start_device()?;

        loop {
            match self.do_write() {
                Ok(LoopState::Stop) => return Ok(()),
                Ok(LoopState::Running) => continue,
                Err(e) => {
                    println!("Recovering output {} after error ({})", self.spec.name, e);
                    self.pcm
                        .recover(e.errno().map(|x| x as i32).unwrap_or(0), true)?;
                    self.start_device()?
                }
            }
        }
    }

    fn run_loop(&mut self) {
        unsafe {
            let sched_param = libc::sched_param { sched_priority: 99 };
            if libc::sched_setscheduler(0, libc::SCHED_RR, &sched_param) < 0 {
                println!("WARNING: Failed to set RR priority for the output thread.");
            }
        }

        match self.run_loop_err() {
            Ok(()) => (),
            Err(e) => println!("Output device {} error: {}", self.spec.name, e),
        }
    }
}

pub struct AlsaOutput {
    spec: DeviceSpec,
    sample_rate: f64,
    min_delay: TimeDelta,
    format: SampleFormat,

    frame_sender: mpsc::Sender<InterleavedFrame>,
    feedback_receiver: mpsc::Receiver<AlsaWriteLoopFeedback>,
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

            // Stop automatic start.
            let swp = pcm.sw_params_current()?;
            swp.set_start_threshold(alsa::pcm::Frames::max_value())?;
            pcm.sw_params(&swp)?
        }

        let device_delay = try!(pcm.delay());
        let min_delay = samples_to_timedelta(sample_rate as f64, device_delay as i64)
            + period_duration * BUFFER_PERIODS as i64 + spec.delay;

        let (sender, receiver) = mpsc::channel();
        let (feedback_sender, feedback_receiver) = mpsc::channel();

        let mut loop_ = AlsaWriteLoop {
            spec: spec.clone(),
            pcm: pcm,
            sample_rate: sample_rate as f64,
            bytes_per_frame: spec.channels.len() * format.bytes_per_sample(),
            period_size: period_size,
            pos_tracker: OutputPosTracker::new(sample_rate as f64),
            rate_detector: RateDetector::new(1.0),

            frame_receiver: receiver,
            feedback_sender: feedback_sender,
            cur_frame: None,
            buffer: vec![],
            buffer_pos: 0,
            last_rate_update: Time::now(),
        };

        thread::spawn(move || {
            loop_.run_loop();
        });

        Ok(AlsaOutput {
            spec: spec,
            sample_rate: sample_rate as f64,
            format: format,
            min_delay: min_delay,
            frame_sender: sender,
            feedback_receiver: feedback_receiver,
        })
    }
}

impl Output for AlsaOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        match self.frame_sender.send(InterleavedFrame {
            timestamp: frame.timestamp,
            buf: frame.to_buffer_with_channel_map(self.format, self.spec.channels.as_slice()),
        }) {
            Ok(()) => (),
            Err(_) => return Err(Error::new("Output was closed.")),
        }

        for msg in self.feedback_receiver.try_iter() {
            match msg {
                AlsaWriteLoopFeedback::CurrentRate(value) => {
                    self.sample_rate = value;
                }
            }
        }

        Ok(())
    }

    fn deactivate(&mut self) {}

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn min_delay(&self) -> TimeDelta {
        self.min_delay
    }
}

const RETRY_PERIOD_SECS: i64 = 3;

pub struct ResilientAlsaOutput {
    spec: DeviceSpec,
    output: Option<AlsaOutput>,
    period_duration: TimeDelta,
    last_open_attempt: Time,
    sample_rate: f64,
    min_delay: TimeDelta,
}

impl ResilientAlsaOutput {
    pub fn new(spec: DeviceSpec, period_duration: TimeDelta) -> Box<Output> {
        let sample_rate = spec.sample_rate.unwrap_or(48000);
        Box::new(ResilientAlsaOutput {
            spec: spec,
            output: None,
            period_duration: period_duration,
            last_open_attempt: Time::now() - TimeDelta::seconds(RETRY_PERIOD_SECS),
            sample_rate: sample_rate as f64,
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
                Err(e) => println!(
                    "INFO: Failed to reopen {} ({}): {}",
                    &self.spec.name, self.spec.id, e
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

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn min_delay(&self) -> TimeDelta {
        self.min_delay
    }
}

pub struct ResamplingOutput {
    output: Box<Output>,
    resampler: resampler::StreamResampler,
}

impl ResamplingOutput {
    pub fn new(output: Box<Output>, window_size: usize) -> Box<Output> {
        let resampler = resampler::StreamResampler::new(output.sample_rate(), window_size);
        Box::new(ResamplingOutput {
            output: output,
            resampler: resampler,
        })
    }
}

impl Output for ResamplingOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        if self.output.sample_rate() == 0.0 {
            return self.output.write(frame);
        }

        self.resampler
            .set_output_sample_rate(self.output.sample_rate());

        match self.resampler.resample(frame) {
            None => Ok(()),
            Some(frame) => self.output.write(frame),
        }
    }

    fn deactivate(&mut self) {
        self.output.deactivate();
    }

    fn sample_rate(&self) -> f64 {
        0.0
    }

    fn min_delay(&self) -> TimeDelta {
        self.output.min_delay() + self.resampler.delay()
    }
}

enum PipeMessage {
    Frame(Frame),
    Deactivate,
}

enum FeedbackMessage {
    SampleRate(f64),
    MinDelay(TimeDelta),
}

pub struct AsyncOutput {
    sender: mpsc::Sender<PipeMessage>,
    feedback_receiver: mpsc::Receiver<FeedbackMessage>,
    sample_rate: f64,
    min_delay: TimeDelta,
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
        };

        thread::spawn(move || {
            let mut sample_rate = 0.0;
            let mut min_delay = TimeDelta::zero();
            loop {
                if sample_rate != output.sample_rate() {
                    sample_rate = output.sample_rate();
                    feedback_sender
                        .send(FeedbackMessage::SampleRate(sample_rate))
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

                let deadline = Time::now() + output.min_delay();
                if frame.timestamp < deadline {
                    println!(
                        "ERROR: Dropping frame: Missed target output time. {:?}",
                        deadline - frame.timestamp
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
                FeedbackMessage::SampleRate(sample_rate) => {
                    self.sample_rate = sample_rate;
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

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn min_delay(&self) -> TimeDelta {
        self.min_delay
    }
}

pub struct CompositeOutput {
    outputs: Vec<(Vec<ChannelPos>, Box<Output>)>,
    num_outs: PerChannel<usize>,
}

impl CompositeOutput {
    pub fn new(
        devices: Vec<DeviceSpec>,
        period_duration: TimeDelta,
        resampler_window: usize,
    ) -> Result<Box<Output>> {
        let mut result = CompositeOutput {
            outputs: vec![],
            num_outs: PerChannel::new(),
        };

        for d in devices {
            let mut channels: Vec<ChannelPos> = vec![];
            for c in d.channels.iter() {
                if !channels.contains(&c) {
                    channels.push(*c);
                    *result.num_outs.get_or_insert(*c, || 0) += 1;
                }
            }

            let out = ResilientAlsaOutput::new(d, period_duration);
            let out = ResamplingOutput::new(out, resampler_window);
            let out = AsyncOutput::new(out);

            result.outputs.push((channels, out));
        }

        Ok(Box::new(result))
    }
}

impl Output for CompositeOutput {
    fn write(&mut self, mut frame: Frame) -> Result<()> {
        if self.outputs.len() == 1 {
            self.outputs[0].1.write(frame)
        } else {
            let mut num_outs = self.num_outs.clone();
            for &mut (ref channels, ref mut out) in self.outputs.iter_mut() {
                let mut out_frame = Frame::new(frame.sample_rate, frame.timestamp, frame.len());
                for c in channels {
                    let no = num_outs.get_mut(*c).unwrap();
                    *no -= 1;
                    if *no > 0 {
                        frame
                            .get_channel(*c)
                            .map(|pcm| out_frame.set_channel(*c, pcm.to_vec()));
                    } else {
                        frame
                            .take_channel(*c)
                            .map(|pcm| out_frame.set_channel(*c, pcm));
                    }
                }
                out.write(out_frame)?;
            }
            Ok(())
        }
    }

    fn deactivate(&mut self) {
        for mut out in self.outputs.iter_mut() {
            out.1.deactivate();
        }
    }

    fn sample_rate(&self) -> f64 {
        0.0
    }

    fn min_delay(&self) -> TimeDelta {
        self.outputs
            .iter()
            .map(|o| o.1.min_delay())
            .max()
            .unwrap_or(TimeDelta::zero())
    }
}
