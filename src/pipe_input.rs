extern crate libc;
extern crate nix;

use std;
use std::fs::File;
use std::io::Read;
use base::*;
use time::{Time, TimeDelta};
use std::os::linux::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use self::nix::fcntl;

use super::input::*;

const FORMAT: SampleFormat = SampleFormat::F32LE;
const FILE_REOPEN_FREQUENCY_SECS: i64 = 2;

pub struct PipeInput {
    spec: DeviceSpec,
    sample_rate: f32,
    file: Option<File>,

    last_open_time: Time,

    period_duration: TimeDelta,

    reference_time: Time,
    pos: i64,

    leftover: Vec<u8>,
}

impl PipeInput {
    pub fn open(spec: DeviceSpec, period_duration: TimeDelta) -> Box<PipeInput> {
        let now = Time::now();
        let sample_rate = spec.sample_rate.unwrap_or(48000) as f32;
        Box::new(PipeInput {
            spec: spec,
            sample_rate: sample_rate,
            file: None,
            last_open_time: now - TimeDelta::seconds(FILE_REOPEN_FREQUENCY_SECS),
            period_duration: period_duration,
            reference_time: now,
            pos: 0,
            leftover: vec![],
        })
    }

    fn try_reopen(&mut self) {
        let now = Time::now();
        if now - self.last_open_time < TimeDelta::seconds(FILE_REOPEN_FREQUENCY_SECS) {
            return;
        }
        self.last_open_time = now;
        match File::open(&self.spec.id) {
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
                    println!("INFO: Opened input: {}", self.spec.id);
                    match fcntl::fcntl(f.as_raw_fd(), fcntl::FcntlArg::F_SETPIPE_SZ(16384)) {
                        Err(e) => println!("WARNING: failed to set buffer size for pipe: {}", e),
                        _ => (),
                    }

                    match fcntl::fcntl(f.as_raw_fd(), fcntl::FcntlArg::F_SETFL(fcntl::OFlag::O_NONBLOCK)) {
                        Err(e) => {
                            println!("ERROR: failed to set O_NONBLOCK: {}", e);
                            return;
                        }
                        _ => (),
                    }

                    self.file = Some(f);
                    self.leftover.clear();
                }
            }
        }
    }
}

impl Input for PipeInput {
    fn read(&mut self) -> Result<Option<Frame>> {
        self.try_reopen();

        let bytes_per_frame = FORMAT.bytes_per_sample() * self.spec.channels.len();

        let size = (self.period_duration * self.sample_rate as i64 / TimeDelta::seconds(1)
            * bytes_per_frame as i64) as usize / 2;
        let mut buffer = vec![0u8; size];

        buffer[0..self.leftover.len()].copy_from_slice(&self.leftover[..]);
        let pos = self.leftover.len();
        let bytes_read = self.leftover.len()
            + match self.file.as_mut().map(|f| f.read(&mut buffer[pos..])) {
                None => {
                    std::thread::sleep(self.period_duration.as_duration());
                    return Ok(None);
                }
                Some(Ok(result)) => result,
                Some(Err(err)) => {
                    if err.raw_os_error() != Some(libc::EAGAIN) {
                        println!("ERROR: read returned: {}", err);
                        self.file = None;
                    }
                    std::thread::sleep(self.period_duration.as_duration());
                    return Ok(None);
                }
            };

        let leftover_bytes = bytes_read % bytes_per_frame;
        let bytes_to_use = bytes_read - leftover_bytes;
        self.leftover = buffer[(buffer.len() - leftover_bytes)..].to_vec();
        if bytes_to_use == 0 {
            std::thread::sleep(self.period_duration.as_duration());
            return Ok(None);
        }

        let mut frame = Frame::from_buffer(
            FORMAT,
            self.sample_rate as f32,
            &self.spec.channels,
            &buffer[0..bytes_to_use],
            get_sample_timestamp(self.reference_time, self.sample_rate, self.pos),
        );

        self.pos += frame.len() as i64;

        let now = Time::now();
        if now - frame.timestamp > self.period_duration * 5 {
            println!(
                "Resetting pipe input, {}",
                (now - frame.timestamp).in_milliseconds()
            );
            self.reference_time = now - self.period_duration;
            self.pos = frame.len() as i64;
            frame.timestamp = self.reference_time;
        }

        std::thread::sleep((frame.timestamp - now).as_duration());

        Ok(Some(frame))
    }

    fn min_delay(&self) -> TimeDelta {
        self.period_duration
    }
}
