extern crate rppal;
extern crate toml;

use base::*;
use gpio;
use std::cmp;
use std::thread;
use std::time::Duration;
use volume_device;

const SLEEP_NS: u64 = 1000;

pub struct PinConfig {
    cs_n: u8,
    sdi: u8,
    sclk: u8,
}

pub struct Pga2311Volume {
    pins: PinConfig,
}

fn parse_pin_param(config: &toml::value::Table, name: &str) -> Result<u8> {
    match config.get(name).and_then(|v| v.as_integer()) {
        Some(v) if v > 0 && v < 30 => Ok(v as u8),
        _ => Err(Error::from_string(format!(
            "PGA2311: Pin {} not specified.",
            name
        ))),
    }
}

fn write_gpio(pin: u8, value: bool) {
    let mut gpio_locked_o = gpio::get_gpio();
    let gpio_locked = gpio_locked_o.as_mut().unwrap();
    gpio_locked.write(
        pin,
        if value {
            rppal::gpio::Level::High
        } else {
            rppal::gpio::Level::Low
        },
    );
}

impl Pga2311Volume {
    pub fn new(pins: PinConfig) -> Result<Pga2311Volume> {
        gpio::initialize()?;

        let mut gpio_locked_o = gpio::get_gpio();
        let gpio_locked = gpio_locked_o.as_mut().unwrap();
        gpio_locked.set_mode(pins.cs_n, rppal::gpio::Mode::Output);
        gpio_locked.set_mode(pins.sdi, rppal::gpio::Mode::Output);
        gpio_locked.set_mode(pins.sclk, rppal::gpio::Mode::Output);
        Ok(Pga2311Volume { pins })
    }

    pub fn create_from_config(
        config: &toml::value::Table,
    ) -> Result<Box<volume_device::VolumeDevice>> {
        Ok(Box::new(Pga2311Volume::new(PinConfig {
            cs_n: parse_pin_param(config, "cs_n")?,
            sdi: parse_pin_param(config, "sdi")?,
            sclk: parse_pin_param(config, "sclk")?,
        })?))
    }

    fn write_bit(&mut self, value: bool) {
        write_gpio(self.pins.sdi, value);
        thread::sleep(Duration::from_nanos(SLEEP_NS));
        write_gpio(self.pins.sclk, true);
        thread::sleep(Duration::from_nanos(SLEEP_NS));
        write_gpio(self.pins.sclk, false);
    }

    fn write_word(&mut self, mut word: u16) {
        write_gpio(self.pins.sclk, false);
        thread::sleep(Duration::from_nanos(SLEEP_NS));
        write_gpio(self.pins.cs_n, false);

        for _ in 0..16 {
            self.write_bit((word & 0x8000) > 0);
            word <<= 1;
        }

        write_gpio(self.pins.cs_n, true)
    }
}

impl volume_device::VolumeDevice for Pga2311Volume {
    fn set_device_gain(&mut self, gain: Gain) -> Gain {
        let v = (192.0 + 2.0 * gain.db).ceil() as i32;
        let v = cmp::max(0, cmp::min(v, 255));
        self.write_word(((v << 8) + v) as u16);
        Gain {
            db: (v - 192) as f32 / 2.0,
        }
    }
}
