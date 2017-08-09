use std::fs::File;
use time::Time;
use std::io;
use base::*;

use super::input::*;

const FORMAT: SampleFormat = SampleFormat::S16LE;
const SAMPLE_RATE: usize = 44100;
const BYTES_PER_SAMPLE: usize = 2 * CHANNELS;

pub struct FileInput {
    file: File,
    start_time: Time,
    pos: i64,
}

impl FileInput {
    pub fn open(filename: &str) -> Result<FileInput> {
        Ok(FileInput {
            file: try!(File::open(filename)),
            start_time: Time::now(),
            pos: 0,
        })
    }
}

impl Input for FileInput {
    fn read(&mut self) -> Result<Option<Frame>> {
        let size = FRAME_SIZE_MS as usize * SAMPLE_RATE / 1000 * BYTES_PER_SAMPLE;
        let mut buffer = vec![0u8; size];
        let bytes_read = try!(io::Read::read(&mut self.file, buffer.as_mut_slice()));
        if bytes_read == 0 {
            return Ok(None);
        }

        let timestamp = get_sample_timestamp(self.start_time, SAMPLE_RATE, self.pos);
        self.pos += (bytes_read / BYTES_PER_SAMPLE) as i64;
        Ok(Some(Frame::from_buffer(
            FORMAT,
            SAMPLE_RATE,
            &buffer[0..bytes_read],
            timestamp,
        )))
    }
}
