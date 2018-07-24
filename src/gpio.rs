extern crate rppal;

use base;
use std::sync::{Mutex, MutexGuard};

lazy_static! {
    static ref G_GPIO: Mutex<Option<rppal::gpio::Gpio>> = Mutex::new(None);
}

pub fn initialize() -> base::Result<()> {
    let mut gpio = G_GPIO.lock().unwrap();

    if gpio.is_some() {
        return Ok(());
    }

    *gpio = Some(rppal::gpio::Gpio::new()?);

    Ok(())
}

pub fn get_gpio() -> MutexGuard<'static, Option<rppal::gpio::Gpio>> {
    G_GPIO.lock().unwrap()
}
