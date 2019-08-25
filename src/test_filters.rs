extern crate byteorder;
extern crate getopts;
extern crate locomix;

use self::byteorder::{NativeEndian, ReadBytesExt};
use getopts::Options;
use locomix::base::*;
use locomix::brutefir::*;
use locomix::filters::*;
use locomix::time;
use std::env;
use std::f64::consts::PI;
use std::fs;

fn get_filter_response<F: StreamFilter>(f: &mut F, sample_rate: f64, freq: f64) -> f64 {
    let mut test_signal = Frame::new(sample_rate, time::Time::now(), 2 * sample_rate as usize);
    {
        let pcm = test_signal.ensure_channel(CHANNEL_FL);
        for i in 0..pcm.len() {
            pcm[i] = (i as f64 / sample_rate * freq * 2.0 * PI).sin() as f32;
        }
    }
    f.reset();
    let response = f.apply(test_signal);

    let mut p_sum = 0.0f64;
    {
        let pcm = response.get_channel(CHANNEL_FL).unwrap();
        for i in (pcm.len() / 2)..pcm.len() {
            p_sum += (pcm[i] as f64).powi(2);
        }
    }

    ((p_sum / ((response.len() / 2) as f64)) * 2.0).log(10.0) * 10.0
}

fn draw_filter_graph<F: StreamFilter>(sample_rate: f64, mut f: F) {
    let mut freq = 20.0f64;
    for _ in 0..82 {
        let response = get_filter_response(&mut f, sample_rate, freq);
        println!("{} {}", freq as usize, response);
        freq = freq * 2.0f64.powf(0.125);
    }
}

fn get_crossfeed_response(sample_rate: f64, freq: f64) -> f64 {
    let mut f = CrossfeedFilter::new(sample_rate);
    let mut test_signal = Frame::new(sample_rate, time::Time::now(), (sample_rate as usize) * 2);
    {
        let pcm = test_signal.ensure_channel(CHANNEL_FL);
        for i in 0..pcm.len() {
            pcm[i] = (i as f64 / sample_rate * freq * 2.0 * PI).sin() as f32;
        }
    }
    {
        let pcm = test_signal.ensure_channel(CHANNEL_FR);
        for i in 0..pcm.len() {
            pcm[i] = (i as f64 / sample_rate * freq * 2.0 * PI).sin() as f32;
        }
    }
    let response = f.apply(test_signal);

    let mut p_sum = 0.0;
    {
        let pcm = response.get_channel(CHANNEL_FL).unwrap();
        for i in 0..response.len() {
            p_sum += (pcm[i] as f64).powi(2);
        }
    }
    ((p_sum / (response.len() as f64)) * 2.0).log(10.0) * 10.0
}

fn draw_crossfeed_graph(sample_rate: f64) {
    let mut freq: f64 = 20.0;
    for _ in 0..82 {
        let response = get_crossfeed_response(sample_rate, freq);
        println!("{} {}", freq as usize, response);
        freq = freq * 2f64.powf(0.125);
    }
}

fn load_fir_params(filename: &str, size: usize) -> Result<FirFilterParams> {
    let mut file = fs::File::open(filename)?;
    let mut result = Vec::<f32>::new();
    loop {
        match file.read_f32::<NativeEndian>() {
            Ok(value) => result.push(value),
            Err(_) => break,
        }
    }
    Ok(FirFilterParams::new(result, size))
}

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optopt("r", "sample-rate", "Internal sample rate", "RATE");
    opts.optmulti("F", "fir", "FIR filter file", "FIR_FILTER");
    opts.optopt(
        "L",
        "filter-length",
        "Length for FIR filter (5000 by default)",
        "FILTER_LENGTH",
    );
    opts.optmulti("b", "biquad", "Biquad filter file", "BIQUAD_FILTER");
    opts.optflag("h", "help", "Print this help menu");

    let matches = match opts.parse(&args[1..]) {
        Err(e) => return Err(Error::from_string(e.to_string())),
        Ok(m) => m,
    };

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    let sample_rate = match matches.opt_str("r").map(|x| x.parse::<usize>()) {
        None => 48000.0,
        Some(Ok(rate)) => rate as f64,
        Some(Err(_)) => return Err(Error::new("Cannot parse sample-rate parameter.")),
    };

    println!("Crossfeed filter");
    draw_crossfeed_graph(sample_rate);

    println!("Bass boost");
    draw_filter_graph(
        sample_rate,
        MultichannelFilter::<BassBoostFilter>::new(SimpleFilterParams::new(sample_rate, 10.0)),
    );

    println!("Loudness filter");
    draw_filter_graph(
        sample_rate,
        MultichannelFilter::<LoudnessFilter>::new(SimpleFilterParams::new(sample_rate, 10.0)),
    );

    let filter_length = match matches.opt_str("L").map(|x| x.parse::<usize>()) {
        None => 5000,
        Some(Ok(length)) => length,
        Some(Err(_)) => return Err(Error::new("Cannot parse filter length.")),
    };

    for filename in matches.opt_strs("F") {
        println!("Brute FIR filter {}", filename);
        let mut f = PerChannel::new();
        f.set(CHANNEL_FL, filename.clone());
        draw_filter_graph(
            sample_rate,
            BruteFir::new(
                f,
                sample_rate as usize,
                time::TimeDelta::milliseconds(5),
                filter_length,
            )?,
        );

        println!("FIR filter {}", filename);
        let mut filters = PerChannel::new();
        filters.set(CHANNEL_FL, load_fir_params(&filename, filter_length)?);
        draw_filter_graph(sample_rate, MultichannelFirFilter::new(filters));
    }

    for filename in matches.opt_strs("b") {
        println!("biquad filter {}", filename);
        let mut filters = PerChannel::new();
        filters.set(CHANNEL_FL, load_biquad_values(&filename)?);
        draw_filter_graph(
            sample_rate,
            PerChannelFilter::<MultiBiquadFilter>::new(filters),
        );
    }

    Ok(())
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
