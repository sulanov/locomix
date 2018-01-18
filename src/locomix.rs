extern crate getopts;
extern crate locomix;
#[macro_use]
extern crate serde_derive;
extern crate toml;

use getopts::Options;
use locomix::alsa_input;
use locomix::async_input;
use locomix::base;
use locomix::control;
use locomix::crossover;
use locomix::filters;
use locomix::input;
use locomix::light;
use locomix::mixer;
use locomix::output;
use locomix::pipe_input;
use locomix::state_script;
use locomix::time::TimeDelta;
use locomix::ui;
use locomix::web_ui;
use std::env;
use std::fs;
use std::io;
use std::io::Read;

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

impl From<io::Error> for RunError {
    fn from(e: io::Error) -> RunError {
        RunError { msg: e.to_string() }
    }
}

impl From<toml::de::Error> for RunError {
    fn from(e: toml::de::Error) -> RunError {
        RunError { msg: e.to_string() }
    }
}

impl From<getopts::Fail> for RunError {
    fn from(e: getopts::Fail) -> RunError {
        RunError { msg: e.to_string() }
    }
}

impl From<base::Error> for RunError {
    fn from(e: base::Error) -> RunError {
        RunError { msg: e.to_string() }
    }
}

#[derive(Deserialize)]
struct InputConfig {
    name: Option<String>,
    #[serde(rename = "type")]
    type_: Option<String>,
    device: String,
    sample_rate: Option<usize>,
    resampler_window: Option<usize>,
    default_gain: Option<f32>,
}

#[derive(Deserialize)]
struct OutputConfig {
    name: Option<String>,
    device: String,
    resampler_window: Option<usize>,
    dynamic_resampling: Option<bool>,
    sample_rate: Option<usize>,
    channel_map: Option<String>,
    fir_filters: Option<Vec<String>>,
    fir_length: Option<usize>,
}

#[derive(Deserialize)]
struct ControlDeviceConfig {
    #[serde(rename = "type")]
    type_: String,
    device: String,
}

#[derive(Deserialize)]
struct Config {
    sample_rate: Option<usize>,
    web_address: Option<String>,
    period_duration: Option<usize>,
    state_script: Option<String>,

    input: Vec<InputConfig>,
    output: Vec<OutputConfig>,
    control_device: Option<Vec<ControlDeviceConfig>>,
}

fn get_resampler_window(value: Option<usize>, dynamic: bool) -> Result<usize, RunError> {
    match (value, dynamic) {
        (None, false) => Ok(100),
        (None, true) => Ok(25),
        (Some(w), _) if w >= 2 && w < 2000 => Ok(w),
        (Some(w), _) => Err(RunError::new(
            format!("Invalid resampler_window: {}", w).as_str(),
        )),
    }
}

fn parse_channel_map(map_str: &str) -> Result<Vec<base::ChannelPos>, RunError> {
    let mut unrecognized = 0;
    let result: Vec<base::ChannelPos> = map_str
        .split_whitespace()
        .map(|c| match c.to_uppercase().as_str() {
            "L" | "FL" | "LEFT" => base::ChannelPos::FL,
            "R" | "FR" | "RIGHT" => base::ChannelPos::FR,
            "S" | "SUB" | "LFE" => base::ChannelPos::Sub,
            "_" => base::ChannelPos::Other,
            _ => {
                unrecognized += 1;
                println!("WARNING: Unrecognized channel name {}", c);
                base::ChannelPos::Other
            }
        })
        .collect();

    if result.len() < 2 || unrecognized == result.len() {
        return Err(RunError::new(
            format!("Invalid channel map: {}", map_str).as_str(),
        ));
    }

    Ok(result)
}

fn run() -> Result<(), RunError> {
    let args: Vec<String> = env::args().collect();

    let mut opts = Options::new();
    opts.optmulti("c", "config", "Config file", "CONFIG_FILE");
    opts.optflag("h", "help", "Print this help menu");

    let matches = try!(opts.parse(&args[1..]));

    if matches.opt_present("h") {
        print!("Usage: {} --config <CONFIG_FILE>\n", args[0]);
        return Ok(());
    }

    let config_filename = match matches.opt_str("c") {
        None => return Err(RunError::new("--config is not specified.")),
        Some(c) => c,
    };

    let mut file = try!(fs::File::open(config_filename));
    let mut config_content = String::new();
    try!(file.read_to_string(&mut config_content));

    let config: Config = try!(toml::from_str(config_content.as_str()));

    let sample_rate = config.sample_rate.unwrap_or(48000);
    if sample_rate < 8000 || sample_rate > 192000 {
        return Err(RunError::new(
            format!("Invalid sample_rate: {}", sample_rate).as_str(),
        ));
    }

    let period_duration = TimeDelta::milliseconds(config.period_duration.unwrap_or(100) as i64);
    if period_duration < TimeDelta::milliseconds(1) || period_duration > TimeDelta::seconds(1) {
        return Err(RunError::new(
            format!(
                "Invalid period_duration: {}",
                config.period_duration.unwrap()
            ).as_str(),
        ));
    }

    let shared_state = ui::SharedState::new();

    let mut inputs = Vec::<async_input::AsyncInput>::new();
    for input in config.input {
        let name = input.name.unwrap_or(input.device.clone());
        let default_gain = input.default_gain.unwrap_or(0.0);
        shared_state.lock().add_input(ui::InputState {
            name: name.clone(),
            gain: ui::Gain { db: default_gain },
        });

        let type_ = input.type_.unwrap_or("alsa".to_string());
        let device: Box<input::Input> = match type_.as_str() {
            "pipe" => pipe_input::PipeInput::open(input.device.as_str(), period_duration),
            "alsa" => alsa_input::ResilientAlsaInput::new(
                base::DeviceSpec {
                    name: name,
                    id: input.device,
                    sample_rate: input.sample_rate,
                    channels: vec![],
                },
                period_duration,
            ),
            _ => {
                return Err(RunError::new(
                    format!("Unknown input type: {}", type_).as_str(),
                ))
            }
        };

        let async_resampled = async_input::AsyncInput::new(Box::new(input::InputResampler::new(
            device,
            sample_rate,
            try!(get_resampler_window(input.resampler_window, false)),
        )));

        inputs.push(async_resampled);
    }

    if inputs.is_empty() {
        return Err(RunError::new("No inputs specified."));
    }

    let mut outputs = Vec::<Box<output::Output>>::new();

    for output in config.output {
        let name = output.name.unwrap_or(output.device.clone());
        shared_state
            .lock()
            .add_output(ui::OutputState::new(name.as_str()));

        let channel_map = match output.channel_map {
            Some(map) => try!(parse_channel_map(map.as_str())),
            None => vec![base::ChannelPos::FL, base::ChannelPos::FR],
        };
        let spec = base::DeviceSpec {
            name: name,
            id: output.device,
            sample_rate: output.sample_rate,
            channels: channel_map,
        };
        let with_subwoofer = spec.channels.iter().any(|c| *c == base::ChannelPos::Sub);
        let out = output::AsyncOutput::new(output::ResilientAlsaOutput::new(spec, period_duration));

        let dynamic_resampling = output.dynamic_resampling.unwrap_or(false);
        let resampler_window = try!(get_resampler_window(
            output.resampler_window,
            dynamic_resampling
        ));
        let out = if dynamic_resampling {
            output::FineResamplingOutput::new(out, resampler_window)
        } else {
            output::ResamplingOutput::new(out, resampler_window)
        };

        let out = if with_subwoofer {
            crossover::SubwooferCrossoverOutput::new(out, &shared_state)
        } else {
            out
        };

        let out = match output.fir_filters.as_ref() {
            None => out,
            Some(files) if files.len() == 2 => {
                let filter_length = output.fir_length.unwrap_or(5000);
                let left = try!(filters::FirFilterParams::new(&files[0]));
                let right = try!(filters::FirFilterParams::new(&files[1]));
                let (left, right) = filters::reduce_fir_pair(left, right, filter_length);
                output::AsyncOutput::new(mixer::FilteredOutput::new(
                    out,
                    filters::MultichannelFirFilter::new_pair(left, right),
                    &shared_state,
                ))
            }
            _ => return Err(RunError::new("fir_filters must contain 2 elements.")),
        };

        outputs.push(output::AsyncOutput::new(out));
    }

    if config.control_device.is_some() {
        for device in config.control_device.unwrap() {
            match device.type_.as_str() {
                "input" => {
                    control::start_input_handler(&device.device, shared_state.clone());
                }
                "griffin_powermate" => {
                    control::start_input_handler(&device.device, shared_state.clone());
                    light::start_light_controller(&device.device, shared_state.clone());
                }
                c => {
                    println!("WARNING: Unrecognized control device type {}", c);
                }
            }
        }
    }

    let web_addr = config.web_address.unwrap_or("0.0.0.0:8000".to_string());
    web_ui::start_web(&web_addr, shared_state.clone());

    match config.state_script {
        Some(s) => state_script::start_state_script_contoller(&s, shared_state.clone()),
        None => (),
    }

    Ok(try!(mixer::run_mixer_loop(
        inputs,
        outputs,
        sample_rate,
        period_duration,
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
