extern crate nix;
extern crate libc;

use std;
use std::fs::File;
use std::io::Read;
use std::time::{Instant, Duration};
use base::*;
use std::os::linux::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use self::nix::fcntl;

use super::input::*;

const FORMAT: SampleFormat = SampleFormat::S32LE;
const SAMPLE_RATE: usize = 44100;
const FILE_REOPEN_FREQUENCY_SECS: u64 = 3;

pub struct PipeInput {
    filename: String,
    file: Option<File>,

    last_open_time: Instant,
}

impl PipeInput {
    pub fn open(filename: &str) -> PipeInput {
        PipeInput {
            filename: String::from(filename),
            file: None,
            last_open_time: Instant::now() - Duration::from_secs(FILE_REOPEN_FREQUENCY_SECS),
        }
    }

    fn try_reopen(&mut self) {
        if self.last_open_time.elapsed() < Duration::from_secs(FILE_REOPEN_FREQUENCY_SECS) {
            return;
        }
        self.last_open_time = Instant::now();
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
        let size = FRAME_SIZE_APPROX_MS * SAMPLE_RATE / 1000 * FORMAT.bytes_per_sample() * CHANNELS;
        let mut buffer = vec![0u8; size];
        const EWOULDBLOCK: i32 = libc::EWOULDBLOCK as i32;
        let bytes_read = match self.file.as_mut().map(|f| f.read(&mut buffer)) {
            None | Some(Ok(0)) => {
                std::thread::sleep(Duration::from_millis(FRAME_SIZE_APPROX_MS as u64));
                return Ok(None);
            }
            Some(Ok(result)) => result,
            Some(Err(err)) => {
                match err.raw_os_error() {
                    Some(EWOULDBLOCK) => {
                        std::thread::sleep(Duration::from_millis(FRAME_SIZE_APPROX_MS as u64))
                    }
                    _ => {
                        println!("ERROR: read returned: {}", err);
                        self.file = None;
                    }
                }
                return Ok(None);
            }
        };

        Ok(Some(Frame::from_buffer(FORMAT, SAMPLE_RATE, &buffer[0..bytes_read])))
    }

    fn samples_buffered(&mut self) -> Result<usize> {
        Ok(0)
    }

    fn is_synchronized(&self) -> bool {
        false
    }
}
