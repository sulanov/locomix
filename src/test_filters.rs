extern crate getopts;
extern crate locomix;

use getopts::Options;
use locomix::base::*;
use locomix::filters::*;
use locomix::time;
use std::f32::consts::PI;
use std::env;

fn get_filter_response<T: AudioFilter<T>>(p: T::Params, sample_rate: FCoef, freq: FCoef) -> FCoef {
    let mut f = T::new(p);
    let mut test_signal = vec![0.0; 2 * sample_rate as usize];
    for i in 0..test_signal.len() {
        test_signal[i] = (i as FCoef / sample_rate * freq * 2.0 * PI).sin() as f32;
    }
    f.apply_multi(&mut test_signal);

    let mut p_sum = 0.0;
    for i in 0..test_signal.len() {
        p_sum += (test_signal[i] as FCoef).powi(2);
    }

    ((p_sum / (test_signal.len() as FCoef)) * 2.0).log(10.0) * 10.0
}

fn draw_filter_graph<T: AudioFilter<T>>(sample_rate: usize, params: T::Params) {
    let mut freq: FCoef = 20.0;
    for _ in 0..82 {
        let response = get_filter_response::<T>(params.clone(), sample_rate as FCoef, freq);
        println!("{} {}", freq as usize, response);
        freq = freq * (2.0 as FCoef).powf(0.125);
    }
}

fn get_crossfeed_response(sample_rate: FCoef, freq: FCoef) -> FCoef {
    let mut f = CrossfeedFilter::new(sample_rate as usize);
    f.set_params(0.3, 0.3);
    let mut test_signal = Frame::new_stereo(
        sample_rate as usize,
        time::Time::now(),
        (sample_rate as usize) * 2,
    );
    for i in 0..test_signal.len() {
        test_signal.channels[0].pcm[i] = (i as FCoef / sample_rate * freq * 2.0 * PI).sin() as f32;
        test_signal.channels[1].pcm[i] = (i as FCoef / sample_rate * freq * 2.0 * PI).sin() as f32;
    }
    let response = f.apply(test_signal);

    let mut p_sum = 0.0;
    for i in 0..response.len() {
        p_sum += (response.channels[0].pcm[i] as FCoef).powi(2);
    }

    ((p_sum / (response.len() as FCoef)) * 2.0).log(10.0) * 10.0
}

fn draw_crossfeed_graph(sample_rate: usize) {
    let mut freq: FCoef = 20.0;
    for _ in 0..82 {
        let response = get_crossfeed_response(sample_rate as FCoef, freq);
        println!("{} {}", freq as usize, response);
        freq = freq * (2 as FCoef).powf(0.125);
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
    opts.optopt("r", "sample-rate", "Internal sample rate", "RATE");
    opts.optmulti("F", "fir", "FIR filter file", "FIR_FILTER");
    opts.optopt(
        "L",
        "filter-length",
        "Length for FIR filter (5000 by default)",
        "FILTER_LENGTH",
    );
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
        None => 48000,
        Some(Ok(rate)) => rate,
        Some(Err(_)) => return Err(Error::new("Cannot parse sample-rate parameter.")),
    };

    println!("Crossfeed filter");
    draw_crossfeed_graph(sample_rate);

    println!("Loudness filter");
    draw_filter_graph::<LoudnessFilter>(
        sample_rate,
        SimpleFilterParams::new(sample_rate, 10.0),
    );

    let filter_length = match matches.opt_str("L").map(|x| x.parse::<usize>()) {
        None => 5000,
        Some(Ok(length)) => length,
        Some(Err(_)) => return Err(Error::new("Cannot parse filter length.")),
    };

    let mut fir_filters = Vec::new();
    for filename in matches.opt_strs("F") {
        let params = match FirFilterParams::new(&filename) {
            Err(e) => return Err(Error::from_string(e.to_string())),
            Ok(m) => m,
        };
        fir_filters.push(params);
    }

    for i in 0..fir_filters.len() {
        println!("FIR filter {}", i);
        draw_filter_graph::<FirFilter>(
            sample_rate,
            reduce_fir(fir_filters[i].clone(), filter_length),
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
