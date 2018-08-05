#![feature(duration_from_micros)]

extern crate serde;

#[macro_use]
extern crate rouille;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate lazy_static;

pub mod alsa_input;
pub mod async_input;
pub mod base;
pub mod brutefir;
pub mod control;
pub mod crossover;
pub mod downmixer;
pub mod filters;
pub mod gpio;
pub mod input;
pub mod input_device;
pub mod light;
pub mod mixer;
pub mod output;
pub mod pga2311;
pub mod pipe_input;
pub mod resampler;
pub mod rotary_encoder;
pub mod state;
pub mod state_script;
pub mod time;
pub mod ui;
pub mod volume_device;
pub mod web_ui;

mod a52_decoder;
