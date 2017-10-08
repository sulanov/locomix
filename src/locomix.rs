extern crate locomix;
extern crate getopts;

use getopts::Options;
use locomix::alsa_input;
use locomix::async_input;
use locomix::base;
use locomix::control;
use locomix::filters;
use locomix::input;
use locomix::light;
use locomix::output;
use locomix::pipe_input;
use locomix::state_script;
use locomix::time::{Time, TimeDelta};
use locomix::mixer;
use locomix::ui;
use locomix::web_ui;
use std::env;

struct RunError {
    msg: String,
}

impl RunError {
    pub fn new(msg: &str) -> RunError {
        RunError {
            msg: String::from(msg),
        }
    }
}

impl From<getopts::Fail> for RunError {
    fn from(e: getopts::Fail) -> RunError {
        RunError{ msg: e.to_string() }
    }
}

impl From<base::Error> for RunError {
    fn from(e: base::Error) -> RunError {
        RunError{ msg: e.to_string() }
    }
}

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
}

fn get_impulse_response(output: &mut output::Output, input: &mut async_input::AsyncInput) -> Result<Vec<f32>, RunError> {
    let frame_duration = TimeDelta::milliseconds(10);
    let frame_size = output.sample_rate() / 100;
    let mut pos = Time::now();

    // Silence for 100 ms.
    for _ in 0..10 {
        let zero = base::Frame::new(output.sample_rate(), pos, frame_size);
        pos += frame_duration;
        try!(output.write(zero));
    }

    // Impulse at -6db.
    let mut impulse = base::Frame::new(output.sample_rate(), pos, frame_size);
    pos += frame_duration;
    impulse.left[0] = 0.5;
    impulse.right[0] = 0.5;
    try!(output.write(impulse));

    // Silence for 500 ms.
    for _ in 0..50 {
        let frame_size = output.sample_rate() / 100;
        let zero = base::Frame::new(output.sample_rate(), pos, frame_size);
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

fn run() -> Result<(), RunError> {
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
    opts.optflag(
        "g",
        "frequency-response",
        "Print out frequency response for all filters",
    );
    opts.optflag("m", "impulse-response", "Measure impulse response");
    opts.optflag("h", "help", "Print this help menu");

    let matches = try!(opts.parse(&args[1..]));

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    let mut inputs = Vec::<Box<input::Input>>::new();
    let mut input_states = Vec::<ui::InputState>::new();

    let sample_rate = match matches.opt_str("r").map(|x| x.parse::<usize>()) {
        None => 48000,
        Some(Ok(rate)) => rate,
        Some(Err(_)) => return Err(RunError::new("Cannot parse sample-rate parameter.")),
    };

    let resampler_window = match matches.opt_str("R").map(|x| x.parse::<usize>()) {
        None => 100,
        Some(Ok(window)) if window > 0 && window <= 1000 => window,
        _ => return Err(RunError::new("Cannot parse resampler-window parameter.")),
    };

    let period_duration = match matches.opt_str("P").map(|x| x.parse::<usize>()) {
        None => TimeDelta::milliseconds(5),
        Some(Ok(d)) if d > 0 && d <= 100 => TimeDelta::milliseconds(d as i64),
        _ => return Err(RunError::new("Cannot parse period-duration parameter.")),
    };

    let filter_length = match matches.opt_str("L").map(|x| x.parse::<usize>()) {
        None => 1000,
        Some(Ok(length)) => length,
        Some(Err(_)) => return Err(RunError::new("Cannot parse filter length.")),
    };

    let mut fir_filters = Vec::new();
    for filename in matches.opt_strs("F") {
        let params = try!(filters::FirFilterParams::new(&filename));
        fir_filters.push(params);
    }

    if matches.opt_present("g") {
        println!("Crossfeed filter");
        filters::draw_crossfeed_graph(sample_rate);

        println!("Loudness filter");
        filters::draw_filter_graph::<filters::LoudnessFilter>(
            sample_rate,
            filters::SimpleFilterParams::new(sample_rate, 10.0),
        );

        println!("Voice Boost filter");
        filters::draw_filter_graph::<filters::VoiceBoostFilter>(
            sample_rate,
            filters::SimpleFilterParams::new(sample_rate, 10.0),
        );

        for i in 0..fir_filters.len() {
            println!("FIR filter {}", i);
            filters::draw_filter_graph::<filters::FirFilter>(
                sample_rate,
                filters::reduce_fir(fir_filters[i].clone(), filter_length),
            );
        }

        return Ok(());
    }

    if matches.opt_present("m") {
        let mut output = try!(output::AlsaOutput::open(
            try!(base::DeviceSpec::parse(&matches.opt_strs("o")[0])),
            period_duration
        ));

        let mut input = async_input::AsyncInput::new(try!(alsa_input::AlsaInput::open(
            try!(base::DeviceSpec::parse(&matches.opt_strs("i")[0])),
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
        let spec = try!(base::DeviceSpec::parse(&o));
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
        let spec = try!(base::DeviceSpec::parse(&i));
        input_states.push(ui::InputState::new(&spec.name));
        inputs.push(alsa_input::ResilientAlsaInput::new(spec, period_duration));
    }

    if inputs.is_empty() {
        return Err(RunError::new("No inputs specified."));
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
        let (left, right) = filters::reduce_fir_pair(
            fir_filters[0].clone(),
            fir_filters[1].clone(),
            filter_length,
        );
        let filtered_output = output::AsyncOutput::new(mixer::FilteredOutput::new(
            outputs.remove(0),
            filters::ParallelFirFilter::new_pair(left, right),
            &shared_state,
        ));
        outputs.insert(0, filtered_output);
    } else if fir_filters.len() != 0 {
        return Err(RunError::new("Expected 0 or 2 FIR filters"));
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

    Ok(try!(mixer::run_mixer_loop(
        wrapped_inputs,
        outputs,
        sample_rate,
        period_duration,
        resampler_window,
        shared_state,
    )))
}

fn main() {
    match run() {
        Err(e) => {
            println!("{}", e.msg);
            std::process::exit(-1);
        }
        Ok(_) => (),
    }
}
