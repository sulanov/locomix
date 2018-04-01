extern crate libc;

use base::*;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::mem;
use std::os::unix::io::AsRawFd;
use std::slice;

const EVIOCSREP: libc::c_ulong = 0x40084503;

#[allow(dead_code)]
struct KbdRepeat {
    delay: i32,
    period: i32,
}

pub const EV_KEY: u16 = 0x01;
pub const EV_REL: u16 = 0x02;
pub const EV_MSC: u16 = 0x04;

pub const KEY_MUTE: u16 = 113;
pub const KEY_VOLUMEDOWN: u16 = 114;
pub const KEY_VOLUMEUP: u16 = 115;
pub const KEY_MENU: u16 = 127;

pub const REL_DIAL: u16 = 0x07;

pub const BTN_MISC: u16 = 0x100;

pub const MSC_PULSELED: u16 = 0x01;

pub const EV_PRESSED: u32 = 1;
pub const EV_REPEATED: u32 = 2;

#[repr(C, packed)]
pub struct InputEvent {
    pub time: libc::timeval,
    pub type_: u16,
    pub code: u16,
    pub value: u32,
}

impl InputEvent {
    pub fn zero() -> InputEvent {
        InputEvent {
            time: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            type_: 0,
            code: 0,
            value: 0,
        }
    }
}

pub struct InputDevice {
    dev: File,
}

impl InputDevice {
    pub fn new(device_path: &str) -> Result<InputDevice> {
        let f = try!(OpenOptions::new().read(true).write(true).open(device_path));

        Ok(InputDevice { dev: f })
    }

    pub fn set_repeat_rate(&self, delay_ms: i32, period_ms: i32) -> Result<()> {
        unsafe {
            let r = KbdRepeat {
                delay: delay_ms,
                period: period_ms,
            };
            let r = libc::ioctl(self.dev.as_raw_fd(), EVIOCSREP, &r);
            if r != 0 {
                return Err(Error::new(&format!(
                    "ioclt(EVIOCSREP) failed, result={}",
                    r
                )));
            }
        }
        Ok(())
    }

    pub fn read(&mut self) -> Result<InputEvent> {
        let mut event = InputEvent::zero();
        let event_size = mem::size_of::<InputEvent>();
        unsafe {
            let event_slice =
                slice::from_raw_parts_mut(&mut event as *mut _ as *mut u8, event_size);
            try!(self.dev.read_exact(event_slice));
        }
        Ok(event)
    }

    pub fn write(&mut self, mut event: InputEvent) -> Result<()> {
        let event_size = mem::size_of::<InputEvent>();
        unsafe {
            let event_slice =
                slice::from_raw_parts_mut(&mut event as *mut _ as *mut u8, event_size);
            try!(self.dev.write_all(event_slice));
        }
        Ok(())
    }
}
