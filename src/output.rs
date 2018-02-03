extern crate alsa;
extern crate nix;

use base::*;
use std::collections::VecDeque;
use std::collections::BTreeMap;
use std::cmp;
use std::ffi::CString;
use std::sync::mpsc;
use std;
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
const RATE_DETECTION_UPDATE_PERIOD_MS: i64 = 500;
const MAX_TIMESTAMP_DEVIATION_US: i64 = 500;
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
    last_update: Time,
}

impl RateDetector {
    fn new(expected_rate: f64) -> RateDetector {
        return RateDetector {
            history: VecDeque::new(),
            sum: 0,
            expected_rate: expected_rate,
            last_update: Time::now(),
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
        if period < TimeDelta::milliseconds(RATE_DETECTION_MIN_PERIOD_MS)
            || (now - self.last_update) < TimeDelta::milliseconds(RATE_DETECTION_UPDATE_PERIOD_MS)
        {
            return None;
        }

        let samples_played = self.sum - self.history[0].size + avail - self.history[0].avail;
        let current_rate = samples_played as f64 / period.in_seconds_f();

        // Allow 2% deviation from the target.
        if (current_rate - self.expected_rate).abs() / self.expected_rate < 0.02 {
            self.last_update = now;
            Some(current_rate)
        } else {
            None
        }
    }
}

enum AlsaWriteLoopFeedback {
    MeasuredSampleRate(f64),
}

struct AlsaWriteLoop {
    device_name: String,
    pcm: alsa::PCM,
    sample_rate: usize,
    format: SampleFormat,
    channels: Vec<ChannelPos>,
    extra_delay: TimeDelta,
    period_size: usize,
    bytes_per_frame: usize,
    rate_detector: RateDetector,

    frame_receiver: mpsc::Receiver<Frame>,
    feedback_sender: mpsc::Sender<AlsaWriteLoopFeedback>,
    cur_frame: Option<Frame>,
    buffer: Vec<u8>,
    buffer_pos: usize,
}

#[derive(PartialEq)]
enum LoopState {
    Continue,
    Stop,
}

impl AlsaWriteLoop {
    fn next_buffer(&mut self, time: Time, deadline: Time) -> LoopState {
        self.buffer_pos = 0;

        loop {
            if self.cur_frame.is_none() {
                let timeout = deadline - Time::now();
                let timeout_duration =
                    std::time::Duration::from_micros(cmp::max(0, timeout.in_microseconds()) as u64);
                self.cur_frame = match self.frame_receiver.recv_timeout(timeout_duration) {
                    Ok(frame) => Some(frame),
                    Err(mpsc::RecvTimeoutError::Timeout) => None,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return LoopState::Stop,
                };
            }

            if self.cur_frame.is_none() {
                //println!("Empty {}", self.period_size);
                self.buffer = vec![0u8; self.period_size as usize * self.bytes_per_frame];
                break;
            }

            let max_deviation = TimeDelta::microseconds(MAX_TIMESTAMP_DEVIATION_US);
            let frame = self.cur_frame.take().unwrap();
            if frame.timestamp - time > max_deviation {
                let gap_samples = ((frame.timestamp - time) * self.sample_rate as i64
                    / TimeDelta::seconds(1)) as usize;
                let samples_to_fill = cmp::min(self.period_size, gap_samples);
                self.buffer = vec![0u8; samples_to_fill * self.bytes_per_frame];

                // Keep the frame and use it next time next_buffer() is called.
                self.cur_frame = Some(frame);
            } else if time - frame.timestamp > max_deviation {
                let samples_to_skip = ((time - frame.timestamp) * self.sample_rate as i64
                    / TimeDelta::seconds(1)) as usize;
                if samples_to_skip >= frame.len() {
                    // Drop this frame - it's too old
                    continue;
                } else {
                    self.buffer =
                        frame.to_buffer_with_channel_map(self.format, self.channels.as_slice());
                    self.buffer.drain(..samples_to_skip * self.bytes_per_frame);
                }
            } else {
                self.buffer =
                    frame.to_buffer_with_channel_map(self.format, self.channels.as_slice());
            }
            break;
        }

        LoopState::Continue
    }

    fn do_write(&mut self) -> alsa::Result<LoopState> {
        let (avail, delay) = try!(self.pcm.avail_delay());
        let avail = avail as usize;

        if self.buffer_pos >= self.buffer.len() {
            let now = Time::now();
            let time = now + samples_to_timedelta(self.sample_rate, delay) + self.extra_delay;
            let deadline =
                now + samples_to_timedelta(self.sample_rate, (self.period_size - avail) as i64);
            if self.next_buffer(time, deadline) == LoopState::Stop {
                return Ok(LoopState::Stop);
            }
        }

        let frames_written = try!(self.pcm.io().writei(&self.buffer[self.buffer_pos..])) as usize;
        self.buffer_pos += frames_written * self.bytes_per_frame;

        match self.rate_detector
            .update(frames_written, try!(self.pcm.avail_update()) as usize)
        {
            None => (),
            Some(rate) => match self.feedback_sender
                .send(AlsaWriteLoopFeedback::MeasuredSampleRate(rate))
            {
                Err(e) => println!("WARNING: Failed to send feedback message {:?}", e),
                _ => (),
            },
        }

        Ok(LoopState::Continue)
    }

    fn run_loop(&mut self) {
        loop {
            match self.do_write() {
                Ok(LoopState::Stop) => return,
                Ok(LoopState::Continue) => continue,
                Err(e) => {
                    println!("Recovering output {} after error ({})", self.device_name, e);
                    self.rate_detector.reset();
                    match self.pcm
                        .recover(e.errno().unwrap_or(nix::Errno::UnknownErrno) as i32, true)
                    {
                        Ok(()) => (),
                        Err(e) => {
                            println!("snd_pcm_recover() failed for {}: {}", self.device_name, e);
                            return;
                        }
                    }
                }
            }
        }
    }
}

pub struct AlsaOutput {
    sample_rate: usize,
    min_delay: TimeDelta,
    measured_sample_rate: f64,

    frame_sender: mpsc::Sender<Frame>,
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
        }

        let (_, device_delay) = try!(pcm.avail_delay());
        let min_delay = samples_to_timedelta(sample_rate, device_delay)
            + period_duration * (BUFFER_PERIODS as i64 - 1) + spec.delay;

        let (sender, receiver) = mpsc::channel();
        let (feedback_sender, feedback_receiver) = mpsc::channel();

        let mut loop_ = AlsaWriteLoop {
            device_name: spec.id,
            pcm: pcm,
            sample_rate: sample_rate,
            format: format,
            bytes_per_frame: spec.channels.len() * format.bytes_per_sample(),
            channels: spec.channels,
            extra_delay: spec.delay,
            period_size: period_size,
            rate_detector: RateDetector::new(sample_rate as f64),

            frame_receiver: receiver,
            feedback_sender: feedback_sender,
            cur_frame: None,
            buffer: vec![],
            buffer_pos: 0,
        };

        thread::spawn(move || {
            loop_.run_loop();
        });

        Ok(AlsaOutput {
            sample_rate: sample_rate,
            min_delay: min_delay,
            measured_sample_rate: sample_rate as f64,
            frame_sender: sender,
            feedback_receiver: feedback_receiver,
        })
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

        match self.frame_sender.send(frame) {
            Ok(()) => (),
            Err(_) => return Err(Error::new("Output was closed.")),
        }

        for msg in self.feedback_receiver.try_iter() {
            match msg {
                AlsaWriteLoopFeedback::MeasuredSampleRate(value) => {
                    self.measured_sample_rate = value;
                }
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
        let sample_rate = spec.sample_rate.unwrap_or(48000);
        Box::new(ResilientAlsaOutput {
            spec: spec,
            output: None,
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
        let out_sample_rate = self.output.sample_rate();
        if out_sample_rate == 0 || out_sample_rate == frame.sample_rate {
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
        0
    }

    fn min_delay(&self) -> TimeDelta {
        self.output.min_delay() + self.resampler.delay()
    }

    fn measured_sample_rate(&self) -> f64 {
        0.0
    }
}

const RATE_UPDATE_PERIOD_MS: i64 = 1000;

pub struct FineResamplingOutput {
    output: Box<Output>,
    resampler: resampler::FineStreamResampler,
    last_rate_update: Time,
}

impl FineResamplingOutput {
    pub fn new(output: Box<Output>, window_size: usize) -> Box<Output> {
        let resampler = resampler::FineStreamResampler::new(
            output.measured_sample_rate(),
            output.sample_rate(),
            window_size,
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
        0
    }

    fn min_delay(&self) -> TimeDelta {
        self.output.min_delay() + self.resampler.delay()
    }

    fn measured_sample_rate(&self) -> f64 {
        0.0
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

pub struct CompositeOutput {
    outputs: Vec<Box<Output>>,
    channel_map: BTreeMap<ChannelPos, usize>,
}

impl CompositeOutput {
    pub fn new(
        devices: Vec<DeviceSpec>,
        period_duration: TimeDelta,
        resampler_window: usize,
        dynamic_resampling: bool,
    ) -> Result<Box<Output>> {
        let mut result = CompositeOutput {
            outputs: vec![],
            channel_map: BTreeMap::new(),
        };

        for d in devices {
            let index = result.outputs.len();
            for &c in d.channels.iter() {
                result.channel_map.entry(c).or_insert(index);
            }

            let out = ResilientAlsaOutput::new(d, period_duration);
            let out = if dynamic_resampling {
                FineResamplingOutput::new(out, resampler_window)
            } else {
                ResamplingOutput::new(out, resampler_window)
            };

            result.outputs.push(out);
        }

        Ok(Box::new(result))
    }
}

impl Output for CompositeOutput {
    fn write(&mut self, mut frame: Frame) -> Result<()> {
        if self.outputs.len() == 1 {
            self.outputs[0].write(frame)
        } else {
            let mut frames: Vec<Frame> = (0..self.outputs.len())
                .map(|_| Frame::new(frame.sample_rate, frame.timestamp))
                .collect();
            for c in frame.channels.drain(..) {
                self.channel_map
                    .get(&c.pos)
                    .map(|&i| frames[i].channels.push(c));
            }
            let result = (0..frames.len())
                .zip(frames.drain(..))
                .fold(Ok(()), |r, (i, f)| r.and(self.outputs[i].write(f)));
            result
        }
    }

    fn deactivate(&mut self) {
        for mut out in self.outputs.iter_mut() {
            out.deactivate();
        }
    }

    fn sample_rate(&self) -> usize {
        0
    }

    fn min_delay(&self) -> TimeDelta {
        self.outputs
            .iter()
            .map(|o| o.min_delay())
            .max()
            .unwrap_or(TimeDelta::zero())
    }

    fn measured_sample_rate(&self) -> f64 {
        0.0
    }
}
