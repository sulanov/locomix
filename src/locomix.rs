#![feature(allocator_api)]

extern crate byteorder;
extern crate getopts;
extern crate locomix;
#[macro_use]
extern crate serde_derive;
extern crate toml;

use self::byteorder::{NativeEndian, ReadBytesExt};
use getopts::Options;
use locomix::alsa_input;
use locomix::async_input;
use locomix::base;
use locomix::brutefir;
use locomix::control;
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
use std::alloc::System;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io;
use std::io::Read;

#[global_allocator]
static ALLOCATOR: System = System;

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
    channel_map: Option<String>,
    default_gain: Option<f32>,
    enable_a52: Option<bool>,

    // Enables sample rate probing. Useful for sound cards that don't
    // detect input rate, e.g. Creative SB X-Fi HD.
    probe_sample_rate: Option<bool>,
}

#[derive(Deserialize)]
struct CompositeOutputEntry {
    device: String,
    sample_rate: Option<usize>,
    channel_map: Option<String>,
    delay: Option<f64>,
}

#[derive(Deserialize)]
struct OutputConfig {
    name: Option<String>,
    device: Option<String>,
    devices: Option<Vec<CompositeOutputEntry>>,
    channel_map: Option<String>,
    sample_rate: Option<usize>,
    default_gain: Option<f32>,
    subwoofer_crossover_frequency: Option<usize>,
    fir_filters: Option<BTreeMap<String, String>>,
    fir_length: Option<usize>,
    use_brutefir: Option<bool>,
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
    enable_crossfeed: Option<bool>,
    resampler_window: Option<usize>,

    input: Vec<InputConfig>,
    output: Vec<OutputConfig>,
    control_device: Option<Vec<ControlDeviceConfig>>,
}

fn parse_channel_id(id: String) -> Result<base::ChannelPos, RunError> {
    match id.to_uppercase().as_str() {
        "L" | "FL" | "LEFT" => Ok(base::ChannelPos::FL),
        "R" | "FR" | "RIGHT" => Ok(base::ChannelPos::FR),
        "C" | "FC" | "CENTER" | "CENTRE" => Ok(base::ChannelPos::FC),
        "SL" | "SURROUND_LEFT" => Ok(base::ChannelPos::SL),
        "SR" | "SURROUND_RIGHT" => Ok(base::ChannelPos::SR),
        "SC" | "SURROUND" | "SURROUND_CENTER" | "SURROUND_CENTRE" => Ok(base::ChannelPos::SC),
        "S" | "B" | "SUB" | "LFE" => Ok(base::ChannelPos::Sub),
        "_" => Ok(base::ChannelPos::Other),
        _ => Err(RunError::new(
            format!("Invalid channel id: {}", id).as_str(),
        )),
    }
}

fn parse_channel_map(map_str: Option<String>) -> Result<Vec<base::ChannelPos>, RunError> {
    let unwrapped = map_str.unwrap_or("L R".to_string());
    let names: Vec<String> = unwrapped
        .as_str()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let mut result: Vec<base::ChannelPos> = vec![];
    for n in names {
        result.push(parse_channel_id(n)?);
    }

    if result.len() < 2 {
        return Err(RunError::new(
            format!("Invalid channel map: {}", unwrapped).as_str(),
        ));
    }

    Ok(result)
}

fn load_fir_params(filename: &str, size: usize) -> Result<filters::FirFilterParams, RunError> {
    let mut file = try!(fs::File::open(filename));
    let mut result = Vec::<f32>::new();
    loop {
        match file.read_f32::<NativeEndian>() {
            Ok(value) => result.push(value),
            Err(_) => break,
        }
    }
    Ok(filters::FirFilterParams::new(result, size))
}

fn load_fir_set(
    files: Option<BTreeMap<String, String>>,
    length: usize,
    sample_rate: usize,
    period_duration: TimeDelta,
    use_brutefir: bool,
) -> Result<Option<Box<filters::StreamFilter>>, RunError> {
    let mut filters = base::PerChannel::new();
    match files {
        None => return Ok(None),
        Some(map) => for (c, f) in map {
            filters.set(parse_channel_id(c)?, f)
        },
    }

    if use_brutefir {
        Ok(Some(Box::new(brutefir::BruteFir::new(
            filters,
            sample_rate,
            period_duration,
            length,
        )?)))
    } else {
        let mut loaded = base::PerChannel::new();
        for (c, f) in filters.iter() {
            loaded.set(c, load_fir_params(&f, length)?);
        }
        Ok(Some(Box::new(filters::MultichannelFirFilter::new(loaded))))
    }
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

    let resampler_window = config.resampler_window.unwrap_or(24);
    if resampler_window < 2 || resampler_window > 2000 {
        return Err(RunError::new(
            format!("Invalid resampler_window: {}", resampler_window).as_str(),
        ));
    }

    let shared_state = ui::SharedState::new();

    let mut inputs = Vec::<async_input::AsyncInput>::new();
    for input in config.input {
        let name = input.name.unwrap_or(input.device.clone());
        let default_gain = input.default_gain.unwrap_or(0.0);
        shared_state.lock().add_input(ui::InputState {
            name: name.clone(),
            gain: base::Gain { db: default_gain },
        });

        let type_ = input.type_.unwrap_or("alsa".to_string());
        let spec = base::DeviceSpec {
            name: name,
            id: input.device,
            sample_rate: input.sample_rate,
            channels: parse_channel_map(input.channel_map)?,
            delay: TimeDelta::zero(),
            enable_a52: input.enable_a52.unwrap_or(false),
        };
        let device: Box<input::Input> = match type_.as_str() {
            "pipe" => pipe_input::PipeInput::open(spec, period_duration),
            "alsa" => alsa_input::ResilientAlsaInput::new(
                spec,
                period_duration,
                input.probe_sample_rate.unwrap_or(false),
            ),
            _ => {
                return Err(RunError::new(
                    format!("Unknown input type: {}", type_).as_str(),
                ))
            }
        };

        let async_resampled = async_input::AsyncInput::new(Box::new(input::InputResampler::new(
            device,
            sample_rate as f64,
            resampler_window,
        )));

        inputs.push(async_resampled);
    }

    if inputs.is_empty() {
        return Err(RunError::new("No inputs specified."));
    }

    let mut outputs = Vec::<Box<output::Output>>::new();
    let mut index = 0;
    for output in config.output {
        index += 1;
        let name = output.name.unwrap_or(format!("Output {}", index));

        let devices = match (output.device, output.channel_map, output.devices) {
            (Some(device), channel_map, None) => vec![base::DeviceSpec {
                name: name.clone(),
                id: device,
                sample_rate: output.sample_rate,
                channels: try!(parse_channel_map(channel_map)),
                delay: TimeDelta::zero(),
                enable_a52: false,
            }],
            (None, None, Some(devices)) => {
                let mut result = vec![];
                for d in devices {
                    result.push(base::DeviceSpec {
                        name: "".to_string(),
                        id: d.device,
                        sample_rate: d.sample_rate.or(output.sample_rate),
                        channels: try!(parse_channel_map(d.channel_map)),
                        delay: TimeDelta::milliseconds_f(d.delay.unwrap_or(0f64)),
                        enable_a52: false,
                    });
                }
                result
            }
            (Some(_), _, Some(_)) => {
                return Err(RunError::new(
                    "device and devices fields cannot be specified together.",
                ));
            }
            (_, Some(_), Some(_)) => {
                return Err(RunError::new(
                    "channel_map cannot be specified for a composite output device.",
                ));
            }
            (None, _, None) => {
                return Err(RunError::new(
                    "Either device or devices must be specified for each output.",
                ));
            }
        };

        let mut channels = base::PerChannel::new();
        for d in devices.iter() {
            for c in d.channels.iter() {
                if *c != base::ChannelPos::Other {
                    channels.set(*c, true);
                }
            }
        }

        let have_subwoofer = channels.get(base::ChannelPos::Sub) == Some(&true);

        let out = try!(output::CompositeOutput::new(
            devices,
            period_duration,
            resampler_window,
        ));

        let sub_config = match (have_subwoofer, output.subwoofer_crossover_frequency) {
            (false, Some(_)) => {
                return Err(RunError::new(
                    format!(
                "subwoofer_crossover_frequency is set for output {}, which doesn't have subwoofer.",
                name
            ).as_str(),
                ))
            }
            (false, None) => None,
            (true, None) => Some(ui::SubwooferConfig {
                crossover_frequency: 80.0,
            }),
            (true, Some(f)) if f > 20 && f < 1000 => Some(ui::SubwooferConfig {
                crossover_frequency: 80.0,
            }),
            (true, Some(f)) => {
                return Err(RunError::new(
                    format!("Invalid subwoofer_crossover_frequency: {}", f).as_str(),
                ))
            }
        };

        let fir_length = output.fir_length.unwrap_or(5000);
        let use_brutefir = output.use_brutefir.unwrap_or(false);
        let fir_filters = load_fir_set(
            output.fir_filters,
            fir_length,
            sample_rate,
            period_duration,
            use_brutefir,
        )?;

        let default_gain = output
            .default_gain
            .unwrap_or((ui::VOLUME_MIN + ui::VOLUME_MAX) / 2.0);
        shared_state.lock().add_output(ui::OutputState {
            name: name.clone(),
            gain: base::Gain { db: default_gain },
            drc_supported: fir_filters.is_some(),
            subwoofer: sub_config,
        });

        let out = output::AsyncOutput::new(mixer::FilteredOutput::new(
            out,
            channels,
            fir_filters,
            sub_config,
            &shared_state,
        ));

        outputs.push(out);
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

    if config.enable_crossfeed.unwrap_or(false) {
        shared_state
            .lock()
            .set_crossfeed(ui::CrossfeedConfig::default());
    }

    Ok(try!(mixer::run_mixer_loop(
        inputs,
        outputs,
        sample_rate as f64,
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
