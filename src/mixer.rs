use crate::async_input::AsyncInput;
use crate::base::*;
use crate::downmixer;
use crate::filter_expr;
use crate::filters::*;
use crate::output;
use crate::state;
use crate::time::*;
use std::cmp;

struct InputMixer {
    input: AsyncInput,
    current_frame: Frame,
    current_frame_pos: usize,
    gain: Gain,
}

impl InputMixer {
    fn new(input: AsyncInput, gain: Gain) -> InputMixer {
        InputMixer {
            input: input,
            current_frame: Frame::new(1.0, Time::zero(), 0),
            current_frame_pos: 0,
            gain: gain,
        }
    }

    fn set_gain(&mut self, gain: Gain) {
        self.gain = gain;
    }

    fn receive_frame(&mut self, deadline: Time, stream_time: Time) -> Result<bool> {
        loop {
            match self.input.read(deadline - Time::now())? {
                None => return Ok(false),
                Some(frame) => {
                    // If the frame is too old then drop it.
                    if frame.end_timestamp() < stream_time {
                        println!("WARNING: Dropping old frame.");
                        continue;
                    }
                    self.current_frame = frame;
                    self.current_frame_pos = 0;
                    return Ok(true);
                }
            }
        }
    }

    fn min_delay(&self) -> TimeDelta {
        self.input.min_delay()
    }

    fn get_frame(&mut self, deadline: Time, stream_time: Time) -> Result<Option<Frame>> {
        if !self.receive_frame(deadline, stream_time)? {
            return Ok(None);
        }
        let mut frame = Frame::new(1.0, Time::zero(), 0);
        std::mem::swap(&mut frame, &mut self.current_frame);
        self.current_frame_pos = 0;

        frame.gain += self.gain;

        Ok(Some(frame))
    }

    fn mix_into(&mut self, mixed_frame: &mut Frame, deadline: Time) -> Result<bool> {
        let multiplier = self.gain.get_multiplier();
        let mut pos = 0;
        while pos < mixed_frame.len() {
            if self.current_frame_pos >= self.current_frame.len() {
                if !self.receive_frame(deadline, mixed_frame.timestamp)? {
                    return Ok(pos > 0);
                }
            }

            let samples = cmp::min(
                self.current_frame.len() - self.current_frame_pos,
                mixed_frame.len() - pos,
            );

            for (c, src_pcm) in self.current_frame.iter_channels() {
                let dst_pcm = mixed_frame.ensure_channel(c);
                for i in 0..samples {
                    dst_pcm[pos + i] += multiplier * src_pcm[self.current_frame_pos + i];
                }
            }

            self.current_frame_pos += samples;
            pos += samples;
        }

        Ok(true)
    }
}

pub struct FilteredOutput {
    output: Box<dyn output::Output>,
    downmixer: downmixer::Downmixer,
    filter_proc: filter_expr::FilterExprProcessor,
}

impl FilteredOutput {
    pub fn new(
        output: Box<dyn output::Output>,
        out_channels: PerChannel<bool>,
        filter_proc: filter_expr::FilterExprProcessor,
    ) -> Box<dyn output::Output> {
        Box::new(FilteredOutput {
            output: output,
            filter_proc,
            downmixer: downmixer::Downmixer::new(out_channels),
        })
    }
}

impl output::Output for FilteredOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        let frame = self.downmixer.process(frame);
        let frame = self.filter_proc.apply(frame);
        self.output.write(frame)
    }

    fn deactivate(&mut self) {
        self.output.deactivate();
        self.filter_proc.reset();
    }

    fn sample_rate(&self) -> f64 {
        self.output.sample_rate()
    }

    fn min_delay(&self) -> TimeDelta {
        self.output.min_delay()
    }
}

const OUTPUT_SHUTDOWN_SECONDS: i64 = 30;
const STANDBY_SECONDS: i64 = 600;

fn get_stream_stats(pcm: Option<&[f32]>) -> state::StreamStats {
    if pcm.is_none() {
        return state::StreamStats {
            rms: 0.0,
            peak: 0.0,
        };
    }
    let pcm = pcm.unwrap();
    let mut sq_sum = 0.0;
    let mut peak = 0.0f32;
    for s in pcm {
        sq_sum += s * s;
        peak = peak.max(s.abs());
    }
    state::StreamStats {
        rms: (2.0 * sq_sum / pcm.len() as f32).sqrt(),
        peak: peak,
    }
}

pub fn run_mixer_loop(
    mut inputs: Vec<AsyncInput>,
    mut outputs: Vec<Box<dyn output::Output>>,
    sample_rate: f64,
    period_duration: TimeDelta,
    shared_state: state::SharedState,
    shared_stream_state: state::SharedStreamInfo,
) -> Result<()> {
    let ui_channel = shared_state.lock().add_observer();

    let mut mixers: Vec<Box<InputMixer>> = {
        let mut pos = 0;
        let state = shared_state.lock();
        inputs
            .drain(..)
            .map(|i| {
                let gain = state.state().inputs[pos].gain;
                pos += 1;
                Box::new(InputMixer::new(i, gain))
            })
            .collect()
    };

    let mut output_gain = shared_state.lock().current_gain();

    let mut last_data_time = Time::now();
    let mut state = state::StreamState::Active;
    let mut selected_output = 0;

    let mut loudness_filter =
        MultichannelFilter::<LoudnessFilter>::new(SimpleFilterParams::new(sample_rate, 10.0));

    let mut bass_boost_filter =
        MultichannelFilter::<BassBoostFilter>::new(SimpleFilterParams::new(sample_rate, 0.0));

    let mut crossfeed_filter = CrossfeedFilter::new(sample_rate);
    crossfeed_filter.set_enabled(
        shared_state
            .lock()
            .state()
            .enable_crossfeed
            .unwrap_or(false),
    );

    let mut exclusive_mux_mode = true;

    let period_size = (period_duration * sample_rate as i64 / TimeDelta::seconds(1)) as usize;

    let mut stream_start_time = Time::now();
    let mut stream_pos: i64 = 0;

    loop {
        let min_input_delay = mixers
            .iter()
            .fold(TimeDelta::zero(), |max, i| cmp::max(max, i.min_delay()));

        let stream_time = get_sample_timestamp(stream_start_time, sample_rate, stream_pos);
        let mix_delay = min_input_delay + period_duration * 2;
        let mix_deadline = stream_time + mix_delay;

        let now = Time::now();
        if now > mix_deadline + period_duration {
            println!("ERROR: Mixer missed deadline. Resetting stream.");
            stream_start_time = now;
            stream_pos = 0;
            continue;
        }

        let mut frame = Frame::new(sample_rate, stream_time, period_size);
        let mut have_data = false;

        if exclusive_mux_mode {
            for m in mixers.iter_mut() {
                match m.get_frame(mix_deadline, stream_time)? {
                    Some(f) => {
                        frame = f;
                        have_data = true;
                        break;
                    }
                    None => (),
                }
            }
            stream_start_time = frame.end_timestamp();
            stream_pos = 0;
        } else {
            for m in mixers.iter_mut() {
                have_data |= m.mix_into(&mut frame, mix_deadline)?;
            }
            stream_pos += frame.len() as i64;
        }

        let old_state = state;

        state = match (have_data, (Time::now() - last_data_time).in_seconds()) {
            (true, _) => {
                last_data_time = now;
                state::StreamState::Active
            }
            (false, t) if t < OUTPUT_SHUTDOWN_SECONDS => state::StreamState::Active,
            (false, t) if t < STANDBY_SECONDS => state::StreamState::Inactive,
            (false, _) => state::StreamState::Standby,
        };

        if state != old_state {
            println!("INFO: state: {}", state.as_str());
            shared_state.lock().set_stream_state(state);
        }

        match state {
            state::StreamState::Active => {
                let stream_info_packet = state::StreamInfoPacket {
                    time: frame.timestamp,
                    left: get_stream_stats(frame.get_channel(CHANNEL_FL)),
                    right: get_stream_stats(frame.get_channel(CHANNEL_FR)),
                };
                {
                    let mut stream_info = shared_stream_state.lock();
                    let expire_time = stream_info_packet.time
                        - TimeDelta::milliseconds(state::STREAM_INFO_PERIOD_MS);
                    while stream_info.packets.len() > 0 && stream_info.packets[0].time < expire_time
                    {
                        stream_info.packets.pop_front();
                    }
                    stream_info.packets.push_back(stream_info_packet);
                }

                frame.gain += output_gain;
                frame = loudness_filter.apply(frame);
                frame = bass_boost_filter.apply(frame);
                frame = crossfeed_filter.apply(frame);

                frame.timestamp +=
                    mix_delay + period_duration * 4 + outputs[selected_output].min_delay();
                outputs[selected_output].write(frame)?;
            }

            _ => {
                if old_state == state::StreamState::Active {
                    // Send an empty frame with gain -100 db to mute the output device. It will
                    // be unmuted when the device is active again.
                    frame.gain = Gain { db: -100.0 };
                    outputs[selected_output].write(frame)?;

                    for ref mut out in &mut outputs {
                        out.deactivate();
                    }
                }

                std::thread::sleep(TimeDelta::milliseconds(500).as_duration());
                stream_start_time = Time::now() - mix_delay;
                stream_pos = 0;
            }
        }

        for msg in ui_channel.try_iter() {
            match msg {
                state::StateChange::SelectOutput { output } => {
                    outputs[selected_output].deactivate();
                    selected_output = output;
                }
                state::StateChange::SetMasterVolume {
                    gain,
                    volume_spl: _,
                    loudness_gain,
                } => {
                    output_gain = gain;
                    loudness_filter
                        .set_params(SimpleFilterParams::new(sample_rate, loudness_gain.db));
                }
                state::StateChange::SetInputGain { device, gain } => {
                    mixers[device as usize].set_gain(gain);
                }
                state::StateChange::SetCrossfeed { enable } => {
                    crossfeed_filter.set_enabled(enable);
                }
                state::StateChange::SetBassBoost { bass_boost } => {
                    bass_boost_filter
                        .set_params(SimpleFilterParams::new(sample_rate, bass_boost.db));
                }
                state::StateChange::SetMuxMode { mux_mode } => {
                    exclusive_mux_mode = mux_mode == state::MuxMode::Exclusive;
                }
                _ => {}
            }
        }
    }
}
