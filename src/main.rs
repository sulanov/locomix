extern crate getopts;
extern crate rustc_serialize;

#[macro_use]
extern crate rouille;

mod alsa_input;
mod async_input;
mod base;
mod control;
mod file_input;
mod filters;
mod input_device;
mod input;
mod light;
mod output;
mod pipe_input;
mod resampler;
mod state_script;
mod ui;
mod web_ui;

use base::*;
use input::Input;
use getopts::Options;
use std::env;
use std::time::{Instant, Duration};
use filters::*;

impl From<getopts::Fail> for Error {
    fn from(e: getopts::Fail) -> Error {
        Error::new(&e.to_string())
    }
}

struct InputMixer {
    input: Box<Input>,
    current_frame: Option<Frame>,
    current_frame_pos: usize,
    volume: ui::Gain,
    gain: ui::Gain,
    multiplier: f32,
}

impl InputMixer {
    fn new(input: Box<Input>) -> InputMixer {
        let mut r = InputMixer {
            input: input,
            current_frame: None,
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

    fn mix_into(&mut self, mixed_frame: &mut Frame) -> Result<bool> {
        for i in 0..mixed_frame.len() {
            if self.current_frame.is_none() {
                self.current_frame = try!(self.input.read());
                self.current_frame_pos = 0;
            }

            let reset: bool;
            match self.current_frame {
                None => return Ok(i > 0),
                Some(ref frame) => {
                    mixed_frame.left[i] += self.multiplier * frame.left[self.current_frame_pos];
                    mixed_frame.right[i] += self.multiplier * frame.right[self.current_frame_pos];
                    self.current_frame_pos += 1;
                    reset = self.current_frame_pos >= frame.len();
                }
            }
            if reset {
                self.current_frame = None;
            }
        }

        if self.input.is_synchronized() {
            while try!(self.input.samples_buffered()) > MAX_QUEUED_FRAMES * mixed_frame.len() {
                let len = try!(self.input.read()).unwrap().len();
                println!("DROPPED {}, left {}",
                         len,
                         try!(self.input.samples_buffered()));
            }
        }

        Ok(true)
    }
}

const OUTPUT_SHUTDOWN_SECONDS: u64 = 5;
const STANDBY_SECONDS: u64 = 3600;

fn run_loop(mut inputs: Vec<Box<Input>>,
            mut outputs: Vec<Box<output::Output>>,
            sample_rate: usize,
            shared_state: ui::SharedState,
            mut drc_filter: Option<StereoFilter<FirFilter>>)
            -> Result<()> {
    let ui_channel = shared_state.lock().add_observer();

    let mut mixers: Vec<Box<InputMixer>> = inputs
        .drain(..)
        .map(|i| Box::new(InputMixer::new(i)))
        .collect();

    let mut state = ui::StreamState::Active;
    let mut last_data_time = Instant::now();
    let mut selected_output = 0;

    let mut loudness_filter =
        StereoFilter::<LoudnessFilter>::new(SimpleFilterParams::new(sample_rate, 10.0));
    let mut voice_boost_filter: Option<StereoFilter<VoiceBoostFilter>> = None;
    let mut crossfeed_filter = CrossfeedFilter::new();

    let mut exclusive_mux_mode = true;
    let mut enable_drc = true;

    loop {
        let mut frame = Frame::new(sample_rate, outputs[selected_output].period_size());

        let mut have_data = false;
        for m in mixers.iter_mut() {
            have_data |= try!(m.mix_into(&mut frame));
            if have_data && exclusive_mux_mode {
                break;
            }
        }

        let now = Instant::now();
        if have_data {
            last_data_time = now;
        }

        let new_state = match (have_data, (now - last_data_time).as_secs()) {
            (true, _) => ui::StreamState::Active,
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

            match (enable_drc, drc_filter.as_mut()) {
                (true, Some(ref mut f)) => {
                    f.apply(&mut frame);
                }
                _ => (),
            }

            try!(outputs[selected_output].write(&frame));
        } else {
            std::thread::sleep(Duration::from_millis(500));
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
                ui::UiMessage::SetEnableDrc { enable } => {
                    enable_drc = enable
                }
                ui::UiMessage::SetVoiceBoost { boost } => {
                    if boost.db > 0.0 {
                        let p = SimpleFilterParams::new(sample_rate, boost.db);
                        if voice_boost_filter.is_some() {
                            voice_boost_filter.as_mut().unwrap().set_params(p);
                        } else {
                            voice_boost_filter = Some(StereoFilter::<VoiceBoostFilter>::new(
                            SimpleFilterParams::new(sample_rate, boost.db)))
                        }
                    } else {
                        voice_boost_filter = None
                    }
                }
                ui::UiMessage::SetCrossfeed { level, delay_ms } => {
                    crossfeed_filter.set_params(level, delay_ms);
                }
                ui::UiMessage::SetMuxMode { mux_mode: ui::MuxMode::Exclusive } => {
                    exclusive_mux_mode = true;
                }
                ui::UiMessage::SetMuxMode { mux_mode: ui::MuxMode::Mixer } => {
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

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optmulti("i", "input", "Audio input device name", "INPUT");
    opts.optmulti("f", "input-file", "Input file name", "FILE");
    opts.optmulti("p", "input-pipe", "Input pipe name", "PIPE");
    opts.optmulti("o", "output", "Output device name", "OUTPUT");
    opts.optopt("r", "sample-rate", "Output sample rate", "RATE");
    opts.optopt("w", "web-address", "Address:port for web UI", "ADDRESS");
    opts.optopt("s",
                "state-script",
                "Script to run on state change",
                "SCRIPT");
    opts.optmulti("c", "control-device", "Control input device", "INPUT_DEV");
    opts.optmulti("l", "light-device", "Light device", "LIGHT_DEV");
    opts.optmulti("F", "filter", "FIR filter file", "FIR_FILTER");
    opts.optopt("L", "filter-length", "Length for FIR filter (1000 by default)", "FILTER_LENGTH");
    opts.optflag("g", "loudness-graph", "Print out loudness graph");
    opts.optflag("h", "help", "Print this help menu");

    let matches = try!(opts.parse(&args[1..]));

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    let mut inputs = Vec::<Box<Input>>::new();
    let mut input_states = Vec::<ui::InputState>::new();

    let output_rate = match matches.opt_str("r").map(|x| x.parse::<usize>()) {
        None => 88200,
        Some(Ok(rate)) => rate,
        Some(Err(_)) => return Err(Error::new("Cannot parse sample-rate parameter.")),
    };

    if matches.opt_present("g") {
        filters::draw_filter_graph::<VoiceBoostFilter>(SimpleFilterParams::new(88100, 10.0));
        return Ok(());
    }

    for f in matches.opt_strs("f") {
        inputs.push(Box::new(try!(file_input::FileInput::open(&f))));
        input_states.push(ui::InputState::new(&f));
    }

    for p in matches.opt_strs("p") {
        inputs.push(Box::new(pipe_input::PipeInput::open(&p)));
        input_states.push(ui::InputState::new(&p));
    }

    for i in matches.opt_strs("i") {
        inputs.push(Box::new(alsa_input::ResilientAlsaInput::new(&i)));
        input_states.push(ui::InputState::new(&i));
    }

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    if inputs.is_empty() {
        return Err(Error::new("No inputs specified."));
    }

    let mut outputs = Vec::<Box<output::Output>>::new();
    let mut output_states = Vec::<ui::OutputState>::new();

    for o in matches.opt_strs("o") {
        outputs.push(Box::new(output::ResilientAlsaOutput::new(&o, output_rate)));
        output_states.push(ui::OutputState::new(&o));
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

    let filter_length = match matches.opt_str("L").map(|x| x.parse::<usize>()) {
        None => 1000,
        Some(Ok(length)) => length,
        Some(Err(_)) => return Err(Error::new("Cannot parse filter length.")),
    };

    let mut fir_filters = Vec::new();
    for filename in matches.opt_strs("F") {
        let params = try!(FirFilterParams::new(&filename));
        fir_filters.push(params.reduce(filter_length));
    }

    let drc_filter = if fir_filters.len() == 0 {
        None
    } else if fir_filters.len() == 2 {
        Some(StereoFilter::<FirFilter>::new_pair(fir_filters[0].clone(), fir_filters[1].clone()))
    } else {
        return Err(Error::new("Expected 0 or 2 FIR filters"))
    };

    let wrapped_inputs = inputs
        .drain(..)
        .map(|input| {
              Box::<async_input::AsyncInput>::new(async_input::AsyncInput::new(
                Box::new(input::InputResampler::new(input, output_rate))))
              as Box<Input>
          })
        .collect();

    run_loop(wrapped_inputs, outputs, output_rate, shared_state, drc_filter)
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
