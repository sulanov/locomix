extern crate rppal;
extern crate toml;

use self::rppal::spi;
use base::*;
use volume_device;

pub struct Pga2311Volume {
    device: spi::Spi,
}

impl From<spi::Error> for Error {
    fn from(e: spi::Error) -> Error {
        Error::from_string(format!("spi error: {}", e))
    }
}

fn parse_param(config: &toml::value::Table, name: &str, min: u8, max: u8) -> Result<u8> {
    match config.get(name).map(|v| (v, v.as_integer())) {
        Some((_, Some(v))) if v >= min as i64 && v <= max as i64 => Ok(v as u8),
        Some((s, _)) => Err(Error::from_string(format!(
            "PGA2311: Invalid value for {}: {}",
            name, s
        ))),
        None => Err(Error::from_string(format!(
            "PGA2311: {} not specified.",
            name
        ))),
    }
}

const SPI_FREQUENCY_HZ: u32 = 32768;

impl Pga2311Volume {
    pub fn new(bus: spi::Bus, slave: spi::SlaveSelect) -> Result<Pga2311Volume> {
        let device = spi::Spi::new(bus, slave, SPI_FREQUENCY_HZ, spi::Mode::Mode0)?;
        Ok(Pga2311Volume { device })
    }

    pub fn create_from_config(
        config: &toml::value::Table,
    ) -> Result<Box<volume_device::VolumeDevice>> {
        let bus = match parse_param(config, "bus", 0, 2)? {
            0 => spi::Bus::Spi0,
            1 => spi::Bus::Spi1,
            2 => spi::Bus::Spi2,
            _ => panic!(),
        };
        let slave = match parse_param(config, "slave", 0, 2)? {
            0 => spi::SlaveSelect::Ss0,
            1 => spi::SlaveSelect::Ss1,
            2 => spi::SlaveSelect::Ss2,
            _ => panic!(),
        };
        Ok(Box::new(Pga2311Volume::new(bus, slave)?))
    }
}

impl volume_device::VolumeDevice for Pga2311Volume {
    fn set_device_gain(&mut self, gain: Gain) -> Gain {
        let v = (192.0 + 2.0 * gain.db).ceil() as i32;
        let v = (0.max(v).min(255)) as u8;
        match self.device.write(&[v, v]) {
            Ok(bytes) => if bytes != 2 {
                println!("ERROR: SPI write returned {}", bytes);
            },
            Err(e) => {
                println!("ERROR: SPI write failed {}", e);
            }
        }

        Gain {
            db: (v as i32 - 192) as f32 / 2.0,
        }
    }
}
