use async_input::AsyncInput;
use base::*;
use time::*;
use filters::*;
use std;
use std::cmp;
use output;
use ui;

struct InputMixer {
    input: AsyncInput,
    current_frame: Frame,
    current_frame_pos: usize,
    volume: ui::Gain,
    gain: ui::Gain,
    multiplier: f32,
}

impl InputMixer {
    fn new(input: AsyncInput) -> InputMixer {
        let mut r = InputMixer {
            input: input,
            current_frame: Frame::new_stereo(1, Time::zero(), 0),
            current_frame_pos: 0,
            volume: ui::Gain { db: -20.0 },
            gain: ui::Gain { db: 0.0 },
            multiplier: 1.0,
        };
        r.update_multiplier();
        r
    }

    fn set_input_gain(&mut self, input_gain: ui::Gain) {
        self.gain = input_gain;
        self.update_multiplier();
    }

    fn set_master_volume(&mut self, volume: ui::Gain) {
        self.volume = volume;
        self.update_multiplier();
    }

    fn update_multiplier(&mut self) {
        self.multiplier = 10f32.powf((self.volume.db + self.gain.db) / 20.0);
    }

    fn mix_into(&mut self, mixed_frame: &mut Frame, deadline: Time) -> Result<bool> {
        let mut pos = 0;
        while pos < mixed_frame.len() {
            if self.current_frame_pos >= self.current_frame.len() {
                match try!(self.input.read(deadline - Time::now())) {
                    None => return Ok(pos > 0),
                    Some(frame) => {
                        // If the frame is too old then drop it.
                        if frame.timestamp < mixed_frame.timestamp {
                            continue;
                        }
                        self.current_frame = frame;
                        self.current_frame_pos = 0;
                    }
                }
            }

            let samples = cmp::min(
                self.current_frame.len() - self.current_frame_pos,
                mixed_frame.len() - pos,
            );

            for mut channel in &mut mixed_frame.channels {
                let src = match self.current_frame
                    .channels
                    .iter()
                    .find(|c| c.pos == channel.pos)
                {
                    Some(c) => c,
                    None => continue,
                };
                for i in 0..samples {
                    channel.pcm[pos + i] += self.multiplier * src.pcm[self.current_frame_pos + i];
                }
            }

            self.current_frame_pos += samples;
            pos += samples;
        }

        Ok(true)
    }
}

pub struct FilteredOutput {
    output: Box<output::Output>,
    filter: MultichannelFirFilter,
    enabled: bool,
    ui_msg_receiver: ui::UiMessageReceiver,
}

impl FilteredOutput {
    pub fn new(
        output: Box<output::Output>,
        filter: MultichannelFirFilter,
        shared_state: &ui::SharedState,
    ) -> Box<output::Output> {
        Box::new(FilteredOutput {
            output: output,
            filter: filter,
            enabled: true,
            ui_msg_receiver: shared_state.lock().add_observer(),
        })
    }
}

impl output::Output for FilteredOutput {
    fn write(&mut self, mut frame: Frame) -> Result<()> {
        for msg in self.ui_msg_receiver.try_iter() {
            match msg {
                ui::UiMessage::SetEnableDrc { enable } => self.enabled = enable,
                _ => (),
            }
        }
        if self.enabled {
            self.filter.apply(&mut frame);
        }
        self.output.write(frame)
    }

    fn deactivate(&mut self) {
        self.output.deactivate();
    }

    fn sample_rate(&self) -> usize {
        self.output.sample_rate()
    }

    fn min_delay(&self) -> TimeDelta {
        self.output.min_delay()
    }

    fn measured_sample_rate(&self) -> f64 {
        self.output.measured_sample_rate()
    }
}

const OUTPUT_SHUTDOWN_SECONDS: i64 = 30;
const STANDBY_SECONDS: i64 = 3600;

pub fn run_mixer_loop(
    mut inputs: Vec<AsyncInput>,
    mut outputs: Vec<Box<output::Output>>,
    sample_rate: usize,
    period_duration: TimeDelta,
    shared_state: ui::SharedState,
) -> Result<()> {
    let ui_channel = shared_state.lock().add_observer();

    let mut mixers: Vec<Box<InputMixer>> = inputs
        .drain(..)
        .map(|i| Box::new(InputMixer::new(i)))
        .collect();

    let mut last_data_time = Time::now();
    let mut state = ui::StreamState::Active;
    let mut selected_output = 0;

    let mut loudness_filter =
        MultichannelFilter::<LoudnessFilter>::new(SimpleFilterParams::new(sample_rate, 10.0));
    let mut crossfeed_filter = CrossfeedFilter::new(sample_rate);

    let mut exclusive_mux_mode = true;

    let min_input_delay = inputs
        .iter()
        .fold(TimeDelta::zero(), |max, i| cmp::max(max, i.min_delay()));
    let mix_delay = min_input_delay + period_duration;

    let period_size = (period_duration * sample_rate as i64 / TimeDelta::seconds(1)) as usize;

    let mut stream_start_time = Time::now() - mix_delay;
    let mut stream_pos: i64 = 0;

    loop {
        let frame_timestamp = get_sample_timestamp(stream_start_time, sample_rate, stream_pos);
        let mut frame = Frame::new_stereo(sample_rate, frame_timestamp, period_size);
        stream_pos += frame.len() as i64;

        let mut have_data = false;
        let mut mix_deadline = frame.end_timestamp() + mix_delay;

        let now = Time::now();
        if now > mix_deadline + period_duration {
            println!(
                "ERROR: Mixer missed deadline. Resetting stream. {:?}",
                now - mix_deadline
            );
            frame.timestamp = frame.end_timestamp();
            stream_pos += frame.len() as i64;
            mix_deadline = frame.end_timestamp() + mix_delay;
        }

        for m in mixers.iter_mut() {
            have_data |= try!(m.mix_into(&mut frame, mix_deadline));
            if have_data && exclusive_mux_mode {
                break;
            }
        }

        let new_state = match (have_data, (Time::now() - last_data_time).in_seconds()) {
            (true, _) => {
                last_data_time = now;
                ui::StreamState::Active
            }
            (false, t) if t < OUTPUT_SHUTDOWN_SECONDS => ui::StreamState::Active,
            (false, t) if t < STANDBY_SECONDS => ui::StreamState::Inactive,
            (false, _) => ui::StreamState::Standby,
        };

        if state != new_state {
            state = new_state;
            println!("INFO: state: {}", state.as_str());
            shared_state.lock().on_stream_state(state);
            if state != ui::StreamState::Active {
                for ref mut out in &mut outputs {
                    out.deactivate();
                }
            }
        }

        if state == ui::StreamState::Active {
            loudness_filter.apply(&mut frame);
            frame = crossfeed_filter.apply(frame);

            frame.timestamp += mix_delay + period_duration * 2 + outputs[selected_output].min_delay();
            try!(outputs[selected_output].write(frame));
        } else {
            std::thread::sleep(TimeDelta::milliseconds(500).as_duration());
            stream_start_time = Time::now() - mix_delay;
            stream_pos = 0;
        }

        for msg in ui_channel.try_iter() {
            match msg {
                ui::UiMessage::SelectOutput { output } => {
                    outputs[output].deactivate();
                    selected_output = output;
                }
                ui::UiMessage::SetMasterVolume { volume, loudness } => {
                    for i in mixers.iter_mut() {
                        i.set_master_volume(volume)
                    }
                    loudness_filter.set_params(SimpleFilterParams::new(sample_rate, loudness.db));
                }
                ui::UiMessage::SetInputGain { device, gain } => {
                    mixers[device as usize].set_input_gain(gain);
                }
                ui::UiMessage::SetEnableDrc { enable: _ } => (),
                ui::UiMessage::SetSubwooferConfig { config: _ } => (),
                ui::UiMessage::SetCrossfeed { level, delay_ms } => {
                    crossfeed_filter.set_params(level, delay_ms);
                }
                ui::UiMessage::SetMuxMode {
                    mux_mode: ui::MuxMode::Exclusive,
                } => {
                    exclusive_mux_mode = true;
                }
                ui::UiMessage::SetMuxMode {
                    mux_mode: ui::MuxMode::Mixer,
                } => {
                    exclusive_mux_mode = false;
                }
            }
        }
    }
}
