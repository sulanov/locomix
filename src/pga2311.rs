extern crate rppal;

use self::rppal::gpio;
use base::*;
use std::cmp;
use std::collections::BTreeMap;
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
    gpio: gpio::Gpio,
    pins: PinConfig,
}

fn parse_pin_param(config: &BTreeMap<String, String>, name: &str) -> Result<u8> {
    match config.get(name) {
        Some(s) => match s.parse::<u8>() {
            Ok(v) => Ok(v),
            Err(_) => Err(Error::from_string(format!(
                "PGA2311: Invalid value for {}: {}",
                name, s
            ))),
        },
        None => Err(Error::from_string(format!(
            "PGA2311: Pin {} not specified.",
            name
        ))),
    }
}

impl Pga2311Volume {
    pub fn new(pins: PinConfig) -> Result<Pga2311Volume> {
        let mut gpio = gpio::Gpio::new()?;
        gpio.set_clear_on_drop(false);
        gpio.set_mode(pins.cs_n, gpio::Mode::Output);
        gpio.set_mode(pins.sdi, gpio::Mode::Output);
        gpio.set_mode(pins.sclk, gpio::Mode::Output);
        Ok(Pga2311Volume { gpio, pins })
    }

    pub fn create_from_config(
        config: &BTreeMap<String, String>,
    ) -> Result<Box<volume_device::VolumeDevice>> {
        Ok(Box::new(Pga2311Volume::new(PinConfig {
            cs_n: parse_pin_param(config, "cs_n")?,
            sdi: parse_pin_param(config, "sdi")?,
            sclk: parse_pin_param(config, "sclk")?,
        })?))
    }

    fn write_bit(&mut self, value: bool) {
        self.gpio.write(
            self.pins.sdi,
            if value {
                gpio::Level::High
            } else {
                gpio::Level::Low
            },
        );
        thread::sleep(Duration::from_nanos(SLEEP_NS));
        self.gpio.write(self.pins.sclk, gpio::Level::High);
        thread::sleep(Duration::from_nanos(SLEEP_NS));
        self.gpio.write(self.pins.sclk, gpio::Level::Low);
    }

    fn write_word(&mut self, mut word: u16) {
        self.gpio.write(self.pins.sclk, gpio::Level::Low);
        thread::sleep(Duration::from_nanos(SLEEP_NS));
        self.gpio.write(self.pins.cs_n, gpio::Level::Low);

        for _ in 0..16 {
            self.write_bit((word & 0x8000) > 0);
            word <<= 1;
        }

        self.gpio.write(self.pins.cs_n, gpio::Level::High)
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
