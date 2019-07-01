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
use locomix::pga2311;
use locomix::pipe_input;
use locomix::rotary_encoder;
use locomix::state;
use locomix::state_script;
use locomix::time::TimeDelta;
use locomix::ui;
use locomix::volume_device;
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
            msg: msg.to_string(),
        }
    }
    pub fn from_string(msg: String) -> RunError {
        RunError { msg }
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
    volume: Option<toml::value::Table>,
}

#[derive(Deserialize)]
struct SpeakersConfig {
    name: String,
    full_scale_spl: Option<f32>,
    sensitivity_1v: Option<f32>,
    sensitivity_1dv: Option<f32>,
    sensitivity_1mw: Option<f32>,
    impedance_ohm: Option<f32>,
}

#[derive(Deserialize)]
struct OutputConfig {
    name: Option<String>,
    devices: Option<Vec<CompositeOutputEntry>>,
    device: Option<String>,
    sample_rate: Option<usize>,
    channel_map: Option<String>,
    volume: Option<toml::value::Table>,

    subwoofer_crossover_frequency: Option<usize>,
    fir_filters: Option<BTreeMap<String, String>>,
    fir_length: Option<usize>,
    use_brutefir: Option<bool>,

    biquad_filters: Option<BTreeMap<String, String>>,

    full_scale_output_volts: Option<f32>,
    speakers: Option<Vec<SpeakersConfig>>,
}

#[derive(Deserialize)]
struct Config {
    sample_rate: Option<usize>,
    static_rate: Option<bool>,
    web_address: Option<String>,
    period_duration: Option<usize>,
    state_script: Option<String>,
    enable_crossfeed: Option<bool>,
    resampler_window: Option<usize>,

    input: Vec<InputConfig>,
    output: Vec<OutputConfig>,
    control_device: Option<Vec<toml::value::Table>>,
    enable_display: Option<bool>,
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

fn load_fir_filters(
    files: Option<BTreeMap<String, String>>,
    length: usize,
    sample_rate: usize,
    period_duration: TimeDelta,
    use_brutefir: bool,
) -> Result<Option<Box<dyn filters::StreamFilter>>, RunError> {
    let mut filters = base::PerChannel::new();
    match files {
        None => return Ok(None),
        Some(map) => {
            for (c, f) in map {
                filters.set(parse_channel_id(c)?, f)
            }
        }
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

fn load_biquad_filters(
    files: Option<BTreeMap<String, String>>,
) -> Result<Option<Box<dyn filters::StreamFilter>>, RunError> {
    let mut filters = base::PerChannel::new();
    match files {
        None => return Ok(None),
        Some(map) => {
            for (c, f) in map {
                filters.set(parse_channel_id(c)?, filters::load_biquad_config(&f)?)
            }
        }
    }

    Ok(Some(Box::new(filters::PerChannelFilter::<
        filters::MultiBiquadFilter,
    >::new(filters))))
}

fn process_speaker_config(
    full_scale_output_volts: f32,
    c: SpeakersConfig,
) -> Result<state::SpeakersConfig, RunError> {
    let fs_output = match (
        c.full_scale_spl,
        c.sensitivity_1v,
        c.sensitivity_1dv,
        c.sensitivity_1mw,
        c.impedance_ohm,
    ) {
        (Some(full_scale_spl), _, _, _, _) => full_scale_spl,
        (_, Some(sensitivity_1v), _, _, _) => {
            sensitivity_1v + 20.0 * full_scale_output_volts.log(10.0)
        }
        (_, _, Some(sensitivity_1dv), _, _) => {
            sensitivity_1dv + 20.0 + 20.0 * full_scale_output_volts.log(10.0)
        }
        (_, _, _, Some(sensitivity_1mw), Some(impedance_ohm)) => {
            sensitivity_1mw
                + 10.0 * (full_scale_output_volts.powi(2) / impedance_ohm / 0.001).log(10.0)
        }
        (_, _, _, None, Some(_)) | (_, _, _, Some(_), None) => return Err(RunError::new(
            "Both sensitivity_1mw and c.impedance_ohm must be specified to calculate output level",
        )),
        _ => {
            return Err(RunError::new(
                format!("Can't calculate output level for speakers {}", c.name).as_str(),
            ))
        }
    };

    Ok(state::SpeakersConfig {
        name: c.name,
        full_scale_spl: fs_output,
    })
}

fn create_volume_device(
    config: Option<toml::value::Table>,
) -> base::Result<Option<Box<dyn volume_device::VolumeDevice>>> {
    let dict = match config {
        Some(d) => d,
        None => return Ok(None),
    };

    let type_ = match dict.get("type") {
        Some(t) => t,
        None => return Err(base::Error::new("Volume device type is missing")),
    };

    let type_str = match type_.as_str() {
        Some(s) => s,
        None => return Err(base::Error::new("Volume device type must be a string")),
    };

    match type_str {
        "alsa" => Ok(Some(volume_device::AlsaVolume::create_from_config(&dict)?)),
        "pga2311" => Ok(Some(pga2311::Pga2311Volume::create_from_config(&dict)?)),
        _ => Err(base::Error::from_string(format!(
            "Unknown volume device type: {}",
            type_
        ))),
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

    let static_rate = config.static_rate.unwrap_or(false);

    let period_duration = TimeDelta::milliseconds(config.period_duration.unwrap_or(100) as i64);
    if period_duration < TimeDelta::milliseconds(1) || period_duration > TimeDelta::seconds(1) {
        return Err(RunError::new(
            format!(
                "Invalid period_duration: {}",
                config.period_duration.unwrap()
            )
            .as_str(),
        ));
    }

    let resampler_window = config.resampler_window.unwrap_or(24);
    if resampler_window < 2 || resampler_window > 2000 {
        return Err(RunError::new(
            format!("Invalid resampler_window: {}", resampler_window).as_str(),
        ));
    }

    let shared_state = state::SharedState::new();

    let mut inputs = Vec::<async_input::AsyncInput>::new();
    for input in config.input {
        let name = input.name.unwrap_or(input.device.clone());
        let default_gain = input.default_gain.unwrap_or(0.0);
        shared_state.lock().add_input(state::InputState {
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
        let device: Box<dyn input::Input> = match type_.as_str() {
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

    let mut outputs = Vec::<Box<dyn output::Output>>::new();
    let mut index = 0;
    for output in config.output {
        index += 1;
        let name = output.name.unwrap_or(format!("Output {}", index));

        let devices = match (output.device, output.channel_map, output.devices) {
            (Some(device), channel_map, None) => vec![(
                base::DeviceSpec {
                    name: name.clone(),
                    id: device,
                    sample_rate: output.sample_rate,
                    channels: try!(parse_channel_map(channel_map)),
                    delay: TimeDelta::zero(),
                    enable_a52: false,
                },
                create_volume_device(output.volume)?,
            )],
            (None, None, Some(devices)) => {
                let mut result = vec![];
                for d in devices {
                    result.push((
                        base::DeviceSpec {
                            name: "".to_string(),
                            id: d.device,
                            sample_rate: d.sample_rate.or(output.sample_rate),
                            channels: try!(parse_channel_map(d.channel_map)),
                            delay: TimeDelta::milliseconds_f(d.delay.unwrap_or(0f64)),
                            enable_a52: false,
                        },
                        create_volume_device(d.volume)?,
                    ));
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
            for c in d.0.channels.iter() {
                if *c != base::ChannelPos::Other {
                    channels.set(*c, true);
                }
            }
        }
        let have_subwoofer = channels.get(base::ChannelPos::Sub) == Some(&true);

        let mut out = output::CompositeOutput::new();
        for (spec, volume) in devices {
            let channels = spec.channels.clone();
            let out_dev = output::ResilientAlsaOutput::new(spec, period_duration, static_rate);

            let out_dev = match volume {
                Some(vol) => Box::new(volume_device::OutputWithVolumeDevice::new(out_dev, vol)),
                None => out_dev,
            };

            let out_dev = output::ResamplingOutput::new(out_dev, resampler_window);
            let out_dev = output::AsyncOutput::new(out_dev);

            out.add_device(channels, out_dev);
        }

        let sub_config = match (have_subwoofer, output.subwoofer_crossover_frequency) {
            (false, Some(_)) => {
                return Err(RunError::new(
                    format!(
                "subwoofer_crossover_frequency is set for output {}, which doesn't have subwoofer.",
                name
            )
                    .as_str(),
                ))
            }
            (false, None) => None,
            (true, None) => Some(state::SubwooferConfig {
                crossover_frequency: 80.0,
            }),
            (true, Some(f)) if f > 20 && f < 1000 => Some(state::SubwooferConfig {
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
        let fir_filters = load_fir_filters(
            output.fir_filters,
            fir_length,
            sample_rate,
            period_duration,
            use_brutefir,
        )?;

        let biquad_filters = load_biquad_filters(output.biquad_filters)?;

        let speakers = match output.speakers {
            Some(speakers) => {
                let mut result = Vec::new();
                let full_scale_output_volts =
                    output.full_scale_output_volts.ok_or(RunError::new(
                        "full_scale_output_volts must be specified with non-empty speaker config.",
                    ))?;
                for s in speakers {
                    result.push(process_speaker_config(full_scale_output_volts, s)?);
                }
                result
            }
            None => Vec::new(),
        };

        shared_state.lock().add_output(state::OutputState {
            name: name.clone(),
            drc_supported: fir_filters.is_some() || fir_filters.is_some(),
            subwoofer: sub_config,
            current_speakers: if speakers.is_empty() { None } else { Some(0) },
            speakers,
        });

        let out = output::AsyncOutput::new(mixer::FilteredOutput::new(
            Box::new(out),
            channels,
            fir_filters,
            biquad_filters,
            sub_config,
            &shared_state,
        ));

        outputs.push(out);
    }

    let event_pipe = ui::EventPipe::new();

    for device in config.control_device.unwrap_or([].to_vec()) {
        let type_ = match device.get("type").map(|t| t.as_str()) {
            None => return Err(RunError::new("control_device type is not specified.")),
            Some(None) => return Err(RunError::new("control_device type must be a string.")),
            Some(Some(v)) => v,
        };
        match type_ {
            "input" => {
                control::start_input_handler(&device, event_pipe.get_event_sink())?;
            }
            "griffin_powermate" => {
                control::start_input_handler(&device, event_pipe.get_event_sink())?;
                light::start_light_controller(&device, shared_state.clone())?;
            }
            "rotary_encoder" => {
                rotary_encoder::start_rotary_encoder_handler(&device, event_pipe.get_event_sink())?;
            }
            c => {
                return Err(RunError::from_string(format!(
                    "ERROR: Unrecognized control device type {}",
                    c
                )))
            }
        }
    }

    let shared_stream_state = state::SharedStreamInfo::new();

    let user_interface = if config.enable_display.unwrap_or(false) {
        ui::DisplayUi::new(shared_stream_state.clone())?
    } else {
        ui::HeadlessUi::new()
    };
    event_pipe.start_ui_thread(user_interface, shared_state.clone());

    let web_addr = config.web_address.unwrap_or("0.0.0.0:8000".to_string());
    web_ui::start_web(&web_addr, shared_state.clone());

    match config.state_script {
        Some(s) => state_script::start_state_script_contoller(&s, shared_state.clone()),
        None => (),
    }

    config
        .enable_crossfeed
        .map(|enable| shared_state.lock().set_enable_crossfeed(enable));

    Ok(try!(mixer::run_mixer_loop(
        inputs,
        outputs,
        sample_rate as f64,
        period_duration,
        shared_state,
        shared_stream_state,
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
