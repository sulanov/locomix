extern crate getopts;
extern crate rustc_serialize;

#[macro_use]
extern crate rouille;

mod alsa_input;
mod async_input;
mod base;
mod control;
mod filters;
mod input_device;
mod input;
mod light;
mod output;
mod pipe_input;
mod resampler;
mod state_script;
mod time;
mod ui;
mod web_ui;

use base::*;
use async_input::AsyncInput;
use input::Input;
use getopts::Options;
use std::env;
use time::{Time, TimeDelta};
use filters::*;

impl From<getopts::Fail> for Error {
    fn from(e: getopts::Fail) -> Error {
        Error::new(&e.to_string())
    }
}

struct InputMixer {
    input: AsyncInput,
    current_frame: Option<Frame>,
    volume: ui::Gain,
    gain: ui::Gain,
    multiplier: f32,
}

enum MixResult {
    Again { new_pos: usize },
    FrameInFuture,
    Done,
}

impl InputMixer {
    fn new(input: AsyncInput) -> InputMixer {
        let mut r = InputMixer {
            input: input,
            current_frame: None,
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

    fn mix(&self, mixed_frame: &mut Frame, frame: &Frame) -> MixResult {
        if frame.timestamp >= mixed_frame.end_timestamp() {
            // Current frame is in the future.
            return MixResult::FrameInFuture;
        }

        if mixed_frame.timestamp >= frame.end_timestamp() {
            // Current frame is in the past.
            return MixResult::Again { new_pos: 0 };
        }

        let start_time = std::cmp::max(frame.timestamp, mixed_frame.timestamp);
        let end_time = std::cmp::min(frame.end_timestamp(), mixed_frame.end_timestamp());
        assert!(start_time < end_time);

        let mut pos = frame.position_at(start_time);
        let end = mixed_frame.position_at(end_time);

        for i in mixed_frame.position_at(start_time)..end {
            mixed_frame.left[i] += self.multiplier * frame.left[pos];
            mixed_frame.right[i] += self.multiplier * frame.right[pos];
            pos += 1;
        }

        if end == mixed_frame.len() {
            MixResult::Done
        } else {
            MixResult::Again { new_pos: end }
        }
    }

    fn mix_into(&mut self, mixed_frame: &mut Frame, deadline: Time) -> Result<bool> {
        let mut pos = 0;
        while pos < mixed_frame.len() {
            if self.current_frame.is_none() {
                self.current_frame = try!(self.input.read(deadline - Time::now()));
            }

            let mix_result = match self.current_frame {
                None => return Ok(pos > 0),
                Some(ref frame) => self.mix(mixed_frame, frame),
            };

            match mix_result {
                MixResult::Again { new_pos } => {
                    pos = new_pos;
                    self.current_frame = None;
                }
                MixResult::FrameInFuture => {
                    return Ok(pos > 0);
                }
                MixResult::Done => {
                    return Ok(true);
                }
            }
        }

        Ok(true)
    }
}

pub struct FilteredOutput {
    output: Box<output::Output>,
    filter: ParallelFirFilter,
    enabled: bool,
    ui_msg_receiver: ui::UiMessageReceiver,
}

impl FilteredOutput {
    pub fn new(
        output: Box<output::Output>,
        filter: ParallelFirFilter,
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

    fn period_size(&self) -> usize {
        self.output.sample_rate()
    }

    fn measured_sample_rate(&self) -> f64 {
        self.output.measured_sample_rate()
    }
}

const OUTPUT_SHUTDOWN_SECONDS: i64 = 5;
const STANDBY_SECONDS: i64 = 3600;

fn run_loop(
    mut inputs: Vec<AsyncInput>,
    mut outputs: Vec<Box<output::Output>>,
    sample_rate: usize,
    period_duration: TimeDelta,
    resampler_window: usize,
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

    let mut stream_start_time = Time::now();
    let mut stream_pos: i64 = 0;

    let mut loudness_filter =
        StereoFilter::<LoudnessFilter>::new(SimpleFilterParams::new(sample_rate, 10.0));
    let mut voice_boost_filter: Option<StereoFilter<VoiceBoostFilter>> = None;
    let mut crossfeed_filter = CrossfeedFilter::new(sample_rate);

    let mut exclusive_mux_mode = true;

    let period_size = (period_duration * sample_rate as i64 / TimeDelta::seconds(1)) as usize;
    let resampler_delay = TimeDelta::seconds(1) * resampler_window as i64 / sample_rate as i64;
    let mix_delay = period_duration * 3 + resampler_delay;
    let target_output_delay = mix_delay + period_duration * 6 + resampler_delay;

    loop {
        let frame_timestamp =
            base::get_sample_timestamp(stream_start_time, sample_rate, stream_pos);
        let mut frame = Frame::new(sample_rate, frame_timestamp, period_size);
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
            match voice_boost_filter.as_mut() {
                Some(ref mut f) => {
                    f.apply(&mut frame);
                }
                None => (),
            }
            frame = crossfeed_filter.apply(frame);

            frame.timestamp += target_output_delay;

            try!(outputs[selected_output].write(frame));
        } else {
            std::thread::sleep(TimeDelta::milliseconds(500).as_duration());
            stream_start_time = Time::now();
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
                ui::UiMessage::SetVoiceBoost { boost } => if boost.db > 0.0 {
                    let p = SimpleFilterParams::new(sample_rate, boost.db);
                    if voice_boost_filter.is_some() {
                        voice_boost_filter.as_mut().unwrap().set_params(p);
                    } else {
                        voice_boost_filter = Some(StereoFilter::<VoiceBoostFilter>::new(
                            SimpleFilterParams::new(sample_rate, boost.db),
                        ))
                    }
                } else {
                    voice_boost_filter = None
                },
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

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
}

fn get_impulse_response(output: &mut output::Output, input: &mut AsyncInput) -> Result<Vec<f32>> {
    let frame_duration = TimeDelta::milliseconds(10);
    let frame_size = output.sample_rate() / 100;
    let mut pos = Time::now();

    // Silence for 100 ms.
    for _ in 0..10 {
        let zero = Frame::new(output.sample_rate(), pos, frame_size);
        pos += frame_duration;
        try!(output.write(zero));
    }

    // Impulse at -6db.
    let mut impulse = Frame::new(output.sample_rate(), pos, frame_size);
    pos += frame_duration;
    impulse.left[0] = 0.5;
    impulse.right[0] = 0.5;
    try!(output.write(impulse));

    // Silence for 500 ms.
    for _ in 0..50 {
        let frame_size = output.sample_rate() / 100;
        let zero = Frame::new(output.sample_rate(), pos, frame_size);
        pos += frame_duration;
        try!(output.write(zero));
    }

    let mut result = Vec::<f32>::new();
    loop {
        match try!(input.read(TimeDelta::zero())) {
            None => return Ok(result),
            Some(s) => result.extend_from_slice(&s.left),
        };
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optmulti("i", "input", "Audio input device name", "INPUT");
    opts.optmulti("p", "input-pipe", "Input pipe name", "PIPE");
    opts.optmulti("o", "output", "Output device name", "OUTPUT");
    opts.optopt("r", "sample-rate", "Internal sample rate", "RATE");
    opts.optopt(
        "R",
        "resampler-window",
        "Resampler window size",
        "WINDOW_SIZE",
    );
    opts.optopt("P", "period-duration", "Period duration, ms", "RATE");
    opts.optopt("w", "web-address", "Address:port for web UI", "ADDRESS");
    opts.optopt(
        "s",
        "state-script",
        "Script to run on state change",
        "SCRIPT",
    );
    opts.optmulti("c", "control-device", "Control input device", "INPUT_DEV");
    opts.optmulti("l", "light-device", "Light device", "LIGHT_DEV");
    opts.optmulti("F", "filter", "FIR filter file", "FIR_FILTER");
    opts.optopt(
        "L",
        "filter-length",
        "Length for FIR filter (1000 by default)",
        "FILTER_LENGTH",
    );
    opts.optflag(
        "D",
        "dynamic-resampling",
        "Enable dynamic stream resampling to match sample rate.",
    );
    opts.optflag("g", "frequency-response", "Print out frequency response for all filters");
    opts.optflag("m", "impulse-response", "Measure impulse response");
    opts.optflag("h", "help", "Print this help menu");

    let matches = try!(opts.parse(&args[1..]));

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    let mut inputs = Vec::<Box<Input>>::new();
    let mut input_states = Vec::<ui::InputState>::new();

    let sample_rate = match matches.opt_str("r").map(|x| x.parse::<usize>()) {
        None => 48000,
        Some(Ok(rate)) => rate,
        Some(Err(_)) => return Err(Error::new("Cannot parse sample-rate parameter.")),
    };

    let resampler_window = match matches.opt_str("R").map(|x| x.parse::<usize>()) {
        None => 100,
        Some(Ok(window)) if window > 0 && window <= 1000 => window,
        _ => return Err(Error::new("Cannot parse resampler-window parameter.")),
    };

    let period_duration = match matches.opt_str("P").map(|x| x.parse::<usize>()) {
        None => TimeDelta::milliseconds(5),
        Some(Ok(d)) if d > 0 && d <= 100 => TimeDelta::milliseconds(d as i64),
        _ => return Err(Error::new("Cannot parse period-duration parameter.")),
    };

    let filter_length = match matches.opt_str("L").map(|x| x.parse::<usize>()) {
        None => 1000,
        Some(Ok(length)) => length,
        Some(Err(_)) => return Err(Error::new("Cannot parse filter length.")),
    };

    let mut fir_filters = Vec::new();
    for filename in matches.opt_strs("F") {
        let params = try!(FirFilterParams::new(&filename));
        fir_filters.push(params);
    }

    if matches.opt_present("g") {
        println!("Crossfeed filter");
        filters::draw_crossfeed_graph(sample_rate);


        println!("Loudness filter");
        filters::draw_filter_graph::<LoudnessFilter>(
            sample_rate,
            SimpleFilterParams::new(sample_rate, 10.0),
        );

        println!("Voice Boost filter");
        filters::draw_filter_graph::<VoiceBoostFilter>(
            sample_rate,
            SimpleFilterParams::new(sample_rate, 10.0),
        );

        for i in 0..fir_filters.len() {
            println!("FIR filter {}", i);
            filters::draw_filter_graph::<FirFilter>(
                sample_rate,
                filters::reduce_fir(fir_filters[i].clone(), filter_length),
            );
        }

        return Ok(());
    }

    if matches.opt_present("m") {
        let mut output = try!(output::AlsaOutput::open(
            try!(DeviceSpec::parse(&matches.opt_strs("o")[0])),
            period_duration
        ));

        let mut input = async_input::AsyncInput::new(try!(alsa_input::AlsaInput::open(
            try!(DeviceSpec::parse(&matches.opt_strs("i")[0])),
            period_duration,
            false
        )));

        let r = try!(get_impulse_response(&mut output, &mut input));
        let mut s = 0;
        for i in 100..r.len() {
            if r[i].abs() > 0.02 {
                s = i - 100;
                break;
            }
        }

        for i in s..r.len() {
            println!("{} {}", i, r[i]);
        }
        return Ok(());
    }

    let mut outputs = Vec::<Box<output::Output>>::new();
    let mut output_states = Vec::<ui::OutputState>::new();

    let dynamic_resampling = matches.opt_present("D");

    for o in matches.opt_strs("o") {
        let spec = try!(DeviceSpec::parse(&o));
        let state = ui::OutputState::new(&spec.name);
        let out = output::AsyncOutput::new(output::ResilientAlsaOutput::new(spec, period_duration));
        let resampled_out = if dynamic_resampling {
            output::FineResamplingOutput::new(out, resampler_window)
        } else {
            output::ResamplingOutput::new(out, resampler_window)
        };
        outputs.push(output::AsyncOutput::new(resampled_out));
        output_states.push(state);
    }

    for p in matches.opt_strs("p") {
        inputs.push(pipe_input::PipeInput::open(&p, period_duration));
        input_states.push(ui::InputState::new(&p));
    }

    for i in matches.opt_strs("i") {
        let spec = try!(DeviceSpec::parse(&i));
        input_states.push(ui::InputState::new(&spec.name));
        inputs.push(alsa_input::ResilientAlsaInput::new(spec, period_duration));
    }

    if inputs.is_empty() {
        return Err(Error::new("No inputs specified."));
    }

    let shared_state = ui::SharedState::new(input_states, output_states);

    let web_addr = match matches.opt_str("w") {
        Some(addr) => addr,
        None => String::from("127.0.0.1:8000"),
    };
    web_ui::start_web(&web_addr, shared_state.clone());

    for c in matches.opt_strs("c") {
        control::start_input_handler(&c, shared_state.clone());
    }

    for l in matches.opt_strs("l") {
        light::start_light_controller(&l, shared_state.clone());
    }

    for s in matches.opt_strs("s") {
        state_script::start_state_script_contoller(&s, shared_state.clone());
    }

    if fir_filters.len() == 2 {
        let (left, right) = reduce_fir_pair(
            fir_filters[0].clone(),
            fir_filters[1].clone(),
            filter_length,
        );
        let filtered_output = output::AsyncOutput::new(FilteredOutput::new(
            outputs.remove(0),
            ParallelFirFilter::new_pair(left, right),
            &shared_state,
        ));
        outputs.insert(0, filtered_output);
    } else if fir_filters.len() != 0 {
        return Err(Error::new("Expected 0 or 2 FIR filters"));
    }

    let wrapped_inputs = inputs
        .drain(..)
        .map(|input| {
            async_input::AsyncInput::new(Box::new(input::InputResampler::new(
                input,
                sample_rate,
                resampler_window,
            )))
        })
        .collect();

    run_loop(
        wrapped_inputs,
        outputs,
        sample_rate,
        period_duration,
        resampler_window,
        shared_state,
    )
}

fn main() {
    match run() {
        Err(e) => {
            println!("{}", e);
            std::process::exit(-1);
        }
        Ok(_) => (),
    }
}
