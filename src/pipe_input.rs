extern crate nix;
extern crate libc;

use std;
use std::fs::File;
use std::io::Read;
use time::{Time, TimeDelta};
use base::*;
use std::os::linux::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use self::nix::fcntl;

use super::input::*;

const FORMAT: SampleFormat = SampleFormat::S32LE;
const SAMPLE_RATE: usize = 44100;
const FILE_REOPEN_FREQUENCY_SECS: i64 = 3;
const BYTES_PER_SAMPLE: usize = 4 * CHANNELS;

pub struct PipeInput {
    filename: String,
    file: Option<File>,

    last_open_time: Time,

    reference_time: Time,
    pos: i64,
}

impl PipeInput {
    pub fn open(filename: &str) -> PipeInput {
        let now = Time::now();
        PipeInput {
            filename: String::from(filename),
            file: None,
            last_open_time: now - TimeDelta::seconds(FILE_REOPEN_FREQUENCY_SECS),
            reference_time: now,
            pos: 0,
        }
    }

    fn try_reopen(&mut self) {
        let now = Time::now();
        if now - self.last_open_time < TimeDelta::seconds(FILE_REOPEN_FREQUENCY_SECS) {
            return;
        }
        self.last_open_time = now;
        match File::open(&self.filename) {
            Err(_) => {
                self.file = None;
            }
            Ok(f) => {
                // Replace the pipe if we didn't have it before or inode has changed.
                let replace = match (self.file.as_mut().map(|f| f.metadata()), f.metadata()) {
                    (None, _) => true,
                    (Some(Ok(m1)), Ok(m2)) => m1.st_ino() != m2.st_ino(),
                    (_, Err(e)) => {
                        println!("ERROR: stat() failed: {}", e);
                        false
                    }
                    (Some(Err(e)), _) => {
                        println!("ERROR: stat() failed: {}", e);
                        false
                    }
                };
                if replace {
                    println!("INFO: Opened input: {}", self.filename);
                    match fcntl::fcntl(f.as_raw_fd(), fcntl::FcntlArg::F_SETPIPE_SZ(16384)) {
                        Err(e) => println!("WARNING: failed to set buffer size for pipe: {}", e),
                        _ => (),
                    }

                    match fcntl::fcntl(f.as_raw_fd(), fcntl::FcntlArg::F_SETFL(fcntl::O_NONBLOCK)) {
                        Err(e) => {
                            println!("ERROR: failed to set O_NONBLOCK: {}", e);
                            return;
                        }
                        _ => (),
                    }

                    self.file = Some(f);
                }
            }
        }
    }
}

impl Input for PipeInput {
    fn read(&mut self) -> Result<Option<Frame>> {
        self.try_reopen();
        let size = FRAME_SIZE_MS * SAMPLE_RATE / 1000 * BYTES_PER_SAMPLE;
        let mut buffer = vec![0u8; size];
        let bytes_read = match self.file.as_mut().map(|f| f.read(&mut buffer)) {
            None | Some(Ok(0)) => {
                std::thread::sleep(TimeDelta::milliseconds(FRAME_SIZE_MS as i64).as_duration());
                return Ok(None);
            }
            Some(Ok(result)) => result,
            Some(Err(err)) => {
                println!("ERROR: read returned: {}", err);
                self.file = None;
                return Ok(None);
            }
        };

        let mut timestamp = get_sample_timestamp(self.reference_time, SAMPLE_RATE, self.pos);
        self.pos += (bytes_read / BYTES_PER_SAMPLE) as i64;

        let now = Time::now();
        if now - timestamp > TimeDelta::milliseconds(100) {
            self.reference_time = now - TimeDelta::milliseconds(FRAME_SIZE_MS as i64);
            self.pos = 0;
            timestamp = self.reference_time;
        }

        std::thread::sleep((timestamp - now).as_duration());

        Ok(Some(Frame::from_buffer(FORMAT, SAMPLE_RATE, &buffer[0..bytes_read], timestamp)))
    }
}
