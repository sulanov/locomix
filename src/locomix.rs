#[macro_use]
extern crate serde_derive;

use self::byteorder::{NativeEndian, ReadBytesExt};
use byteorder;
use getopts;
use getopts::Options;
use locomix;
use locomix::alsa_input;
use locomix::async_input;
use locomix::base;
use locomix::control;
use locomix::filter_expr;
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
use toml;

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
struct FilterConfig {
    name: String,
    biquad: Option<String>,
    biquad_config: Option<String>,
    biquad_values_file: Option<String>,
    fir_file: Option<String>,
    fir_length: Option<usize>,
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

    filtered_channels: Option<BTreeMap<String, String>>,

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
    filter: Vec<FilterConfig>,
    output: Vec<OutputConfig>,
    control_device: Option<Vec<toml::value::Table>>,
    enable_display: Option<bool>,
}

fn parse_channel_map(
    map_str: Option<String>,
    parse_channel: &dyn Fn(&str) -> Option<base::ChannelPos>,
) -> Result<Vec<base::ChannelPos>, RunError> {
    let unwrapped = map_str.unwrap_or("L R".to_string());
    let names: Vec<String> = unwrapped
        .as_str()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let mut result: Vec<base::ChannelPos> = vec![];
    for n in names {
        let c = parse_channel(&n)
            .ok_or_else(|| RunError::from_string(format!("Unknown channel ID: {}", n)))?;
        result.push(c);
    }

    if result.len() < 2 {
        return Err(RunError::new(
            format!("Invalid channel map: {}", unwrapped).as_str(),
        ));
    }

    Ok(result)
}

fn load_fir_params(filename: &str, size: usize) -> Result<filters::FirFilterParams, RunError> {
    let mut file = fs::File::open(filename)?;
    let mut result = Vec::<f32>::new();
    loop {
        match file.read_f32::<NativeEndian>() {
            Ok(value) => result.push(value),
            Err(_) => break,
        }
    }
    Ok(filters::FirFilterParams::new(result, size))
}

fn load_filters(
    filter_configs: Vec<FilterConfig>,
    sample_rate: f64,
) -> Result<BTreeMap<String, filter_expr::FilterConfig>, RunError> {
    let mut result = BTreeMap::new();
    for fc in filter_configs {
        if result.get(&fc.name).is_some() {
            return Err(RunError::from_string(format!(
                "Duplicaite filter name: {}",
                fc.name
            )));
        }
        let f = match (fc.biquad, fc.biquad_config, fc.biquad_values_file, fc.fir_file) {
            (Some(biquad), None, None, None) => filter_expr::FilterConfig::Biquad(filters::parse_biquad_definition(sample_rate, biquad)?),
            (None, Some(filename), None, None) => filter_expr::FilterConfig::Biquad(filters::load_biquad_config(sample_rate, &filename)?),
            (None, None, Some(filename), None) => filter_expr::FilterConfig::Biquad(filters::load_biquad_values(&filename)?),
            (None, None, None, Some(filename)) => filter_expr::FilterConfig::Fir(load_fir_params(&filename, fc.fir_length.unwrap_or(5000))?),
            (None, None, None, None) => return Err(RunError::new("One of `biquad`, `biquad_file`, `biquad_values_file` `fir_file` must be specified for each filter." )),
            _ => return Err(RunError::new("Only one of `biquad`, `biquad_file`, `biquad_values_file` `fir_file` must be specified for each filter." )),
        };
        result.insert(fc.name, f);
    }
    Ok(result)
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

    let matches = opts.parse(&args[1..])?;

    if matches.opt_present("h") {
        print!("Usage: {} --config <CONFIG_FILE>\n", args[0]);
        return Ok(());
    }

    let config_filename = match matches.opt_str("c") {
        None => return Err(RunError::new("--config is not specified.")),
        Some(c) => c,
    };

    let mut file = fs::File::open(config_filename)?;
    let mut config_content = String::new();
    file.read_to_string(&mut config_content)?;

    let config: Config = toml::from_str(config_content.as_str())?;

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
            channels: parse_channel_map(input.channel_map, &base::parse_channel_id)?,
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

    let filters = load_filters(config.filter, sample_rate as f64)?;

    let mut outputs = Vec::<Box<dyn output::Output>>::new();
    let mut index = 0;
    for output in config.output {
        index += 1;
        let name = output.name.unwrap_or(format!("Output {}", index));

        let mut exprs = vec![];
        for (name, expr) in output.filtered_channels.unwrap_or_else(|| BTreeMap::new()) {
            exprs.push((name, filter_expr::parse_filter_expr(&expr)?));
        }
        let filter_proc = filter_expr::FilterExprProcessor::new(&filters, &exprs)?;

        let devices = match (output.device, output.channel_map, output.devices) {
            (Some(device), channel_map, None) => vec![(
                base::DeviceSpec {
                    name: name.clone(),
                    id: device,
                    sample_rate: output.sample_rate,
                    channels: parse_channel_map(
                        channel_map,
                        &(|id| filter_proc.get_channel_pos(id)),
                    )?,
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
                            channels: parse_channel_map(
                                d.channel_map,
                                &(|id| filter_proc.get_channel_pos(id)),
                            )?,
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
        for (d, _) in devices.iter() {
            for c in d.channels.iter() {
                if *c != base::CHANNEL_UNDEFINED {
                    channels.set(*c, true);
                }
            }
        }
        filter_proc.expand_channel_set(&mut channels);

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
            current_speakers: if speakers.is_empty() { None } else { Some(0) },
            speakers,
        });

        let out = output::AsyncOutput::new(mixer::FilteredOutput::new(
            Box::new(out),
            channels,
            filter_proc,
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

    Ok(mixer::run_mixer_loop(
        inputs,
        outputs,
        sample_rate as f64,
        period_duration,
        shared_state,
        shared_stream_state,
    )?)
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
