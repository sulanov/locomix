use std::fs::File;
use std::io;
use base::*;

use super::input::*;

const FORMAT: SampleFormat = SampleFormat::S16LE;
const SAMPLE_RATE: usize = 44100;

pub struct FileInput {
    file: File,
}

impl FileInput {
    pub fn open(filename: &str) -> Result<FileInput> {
        Ok(FileInput { file: try!(File::open(filename)) })
    }
}

impl Input for FileInput {
    fn read(&mut self) -> Result<Option<Frame>> {
        let size = FRAME_SIZE_APPROX_MS * SAMPLE_RATE / 1000 * FORMAT.bytes_per_sample() * CHANNELS;
        let mut buffer = vec![0u8; size];
        let bytes_read = try!(io::Read::read(&mut self.file, buffer.as_mut_slice()));
        if bytes_read == 0 {
            return Ok(None);
        }

        Ok(Some(Frame::from_buffer(FORMAT, SAMPLE_RATE, &buffer[0..bytes_read])))
    }

    fn samples_buffered(&mut self) -> Result<usize> {
        Ok(0)
    }

    fn is_synchronized(&self) -> bool {
        false
    }
}
