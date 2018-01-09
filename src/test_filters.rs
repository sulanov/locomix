extern crate locomix;
extern crate getopts;

use getopts::Options;
use locomix::filters;
use std::env;

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
}

fn run() -> Result<(), String> {
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
        Err(e) => return Result::Err(e.to_string()),
        Ok(m) => m,
    };

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    let sample_rate = match matches.opt_str("r").map(|x| x.parse::<usize>()) {
        None => 48000,
        Some(Ok(rate)) => rate,
        Some(Err(_)) => return Err("Cannot parse sample-rate parameter.".to_string()),
    };

    println!("Crossfeed filter");
    filters::draw_crossfeed_graph(sample_rate);

    println!("Loudness filter");
    filters::draw_filter_graph::<filters::LoudnessFilter>(
        sample_rate,
        filters::SimpleFilterParams::new(sample_rate, 10.0),
    );

    let filter_length = match matches.opt_str("L").map(|x| x.parse::<usize>()) {
        None => 5000,
        Some(Ok(length)) => length,
        Some(Err(_)) => return Err("Cannot parse filter length.".to_string()),
    };

    let mut fir_filters = Vec::new();
    for filename in matches.opt_strs("F") {
        let params = match filters::FirFilterParams::new(&filename) {
            Err(e) => return Result::Err(e.to_string()),
            Ok(m) => m,
        };
        fir_filters.push(params);
    }

    for i in 0..fir_filters.len() {
        println!("FIR filter {}", i);
        filters::draw_filter_graph::<filters::FirFilter>(
            sample_rate,
            filters::reduce_fir(fir_filters[i].clone(), filter_length),
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
