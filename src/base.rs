extern crate alsa;
extern crate simd;

use self::simd::f32x4;
use std::error;
use std::fmt;
use std::io;
use std::result;
use time::{Time, TimeDelta};

pub const CHANNELS: usize = 2;

#[derive(Clone, Copy)]
pub enum SampleFormat {
    S16LE,
    S24LE3,
    S24LE4,
    S32LE,
}

impl SampleFormat {
    pub fn to_str(&self) -> &'static str {
        match *self {
            SampleFormat::S16LE => "S16LE",
            SampleFormat::S24LE3 => "S24LE3",
            SampleFormat::S24LE4 => "S24LE4",
            SampleFormat::S32LE => "S32LE",
        }
    }

    pub fn bytes_per_sample(&self) -> usize {
        match *self {
            SampleFormat::S16LE => 2,
            SampleFormat::S24LE3 => 3,
            SampleFormat::S24LE4 => 4,
            SampleFormat::S32LE => 4,
        }
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

fn clamp(v: f32) -> f32 {
    if v > 1.0 {
        1.0
    } else if v < -1.0 {
        -1.0
    } else {
        v
    }
}

fn write_sample_s16le(val: f32, buf: &mut Vec<u8>) {
    let ival = (clamp(val) as f64 * 2147483648f64 - 32767.5) as i32 as u32;
    buf.push(((ival & 0x00ff0000) >> 16) as u8);
    buf.push(((ival & 0xff000000) >> 24) as u8);
}

fn write_sample_s24le3(val: f32, buf: &mut Vec<u8>) {
    let ival = (clamp(val) as f64 * 2147483648f64 - 127.5) as i32 as u32;
    buf.push(((ival & 0x0000ff00) >> 8) as u8);
    buf.push(((ival & 0x00ff0000) >> 16) as u8);
    buf.push(((ival & 0xff000000) >> 24) as u8);
}

fn write_sample_s24le4(val: f32, buf: &mut Vec<u8>) {
    let ival = (clamp(val) as f64 * 2147483648f64 - 127.5) as i32 as u32;
    buf.push(((ival & 0x0000ff00) >> 8) as u8);
    buf.push(((ival & 0x00ff0000) >> 16) as u8);
    buf.push(((ival & 0xff000000) >> 24) as u8);
    buf.push(0);
}

fn write_sample_s32le(val: f32, buf: &mut Vec<u8>) {
    let ival = (clamp(val) as f64 * 2147483648f64 - 0.5) as i32 as u32;
    buf.push(((ival & 0x000000ff) >> 0) as u8);
    buf.push(((ival & 0x0000ff00) >> 8) as u8);
    buf.push(((ival & 0x00ff0000) >> 16) as u8);
    buf.push(((ival & 0xff000000) >> 24) as u8);
}


fn read_sample_s16le(buf: &[u8], pos: usize) -> f32 {
    (((((buf[pos + 0] as u32) << 16) | ((buf[pos + 1] as u32) << 24)) as i32 as f64 +
        32767.5) / 2147483648f64) as f32
}

fn read_sample_s24le3(buf: &[u8], pos: usize) -> f32 {
    (((((buf[pos + 0] as u32) << 8) | ((buf[pos + 1] as u32) << 16) |
        ((buf[pos + 2] as u32) << 24)) as i32 as f64 + 127.5) / 2147483648f64) as f32
}

fn read_sample_s24le4(buf: &[u8], pos: usize) -> f32 {
    (((((buf[pos + 0] as u32) << 8) | ((buf[pos + 1] as u32) << 16) |
        ((buf[pos + 2] as u32) << 24)) as i32 as f64 + 127.5) / 2147483648f64) as f32
}

fn read_sample_s32le(buf: &[u8], pos: usize) -> f32 {
    (((((buf[pos + 0] as u32) << 0) | ((buf[pos + 1] as u32) << 8) |
        ((buf[pos + 2] as u32) << 16) | ((buf[pos + 3] as u32) << 24)) as i32 as
        f64 + 0.5) / 2147483648f64) as f32
}

// Fast SIMD-optimized convolution. Optimized for NEON on Raspberry PI 3.
pub fn convolve(v1: &[f32], v2: &[f32]) -> f32 {
    assert!(v1.len() == v2.len());

    let mut sum1 = f32x4::splat(0.0);
    let mut sum2 = f32x4::splat(0.0);

    for i in 0..(v1.len() / 8) {
        let v1_0 = f32x4::load(v1, i * 8);
        let v1_4 = f32x4::load(v1, i * 8 + 4);

        let v2_0 = f32x4::load(v2, i * 8);
        let v2_4 = f32x4::load(v2, i * 8 + 4);

        sum1 = sum1 + v1_0 * v2_0;
        sum2 = sum2 + v1_4 * v2_4;
    }

    let mut pos = (v1.len() / 8) * 8;
    while pos + 4 <= v1.len() {
        sum1 = sum1 + f32x4::load(v1, pos) * f32x4::load(v2, pos);
        pos += 4;
    }

    let mut sum_end = 0.0;
    while pos < v1.len() {
        sum_end += v1[pos] * v2[pos];
        pos += 1;
    }

    sum1.extract(0) + sum1.extract(1) + sum1.extract(2) + sum1.extract(3) +
        sum2.extract(0) + sum2.extract(1) + sum2.extract(2) + sum2.extract(3) + sum_end
}

pub fn get_sample_timestamp(start: Time, sample_rate: usize, sample: i64) -> Time {
    start +
        (TimeDelta::seconds(1) * sample + TimeDelta::nanoseconds(sample_rate as i64 / 2)) /
            sample_rate as i64
}

pub struct Frame {
    pub sample_rate: usize,
    pub timestamp: Time,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl Frame {
    pub fn new(sample_rate: usize, timestamp: Time, samples: usize) -> Frame {
        // Avoid denormal zero.
        let zero = 1e-10f32;
        Frame {
            sample_rate: sample_rate,
            timestamp: timestamp,
            left: vec![zero; samples],
            right: vec![zero; samples],
        }
    }

    pub fn len(&self) -> usize {
        assert!(self.left.len() == self.right.len());
        self.left.len()
    }

    pub fn duration(&self) -> TimeDelta {
        (TimeDelta::seconds(1) * self.len() as i64) / self.sample_rate as i64
    }

    pub fn end_timestamp(&self) -> Time {
        self.timestamp + self.duration()
    }

    pub fn to_buffer(&self, format: SampleFormat) -> Vec<u8> {
        let mut buf = Vec::with_capacity(format.bytes_per_sample() * CHANNELS * self.len());
        match format {
            SampleFormat::S16LE => for i in 0..self.len() {
                write_sample_s16le(self.left[i], &mut buf);
                write_sample_s16le(self.right[i], &mut buf);
            },
            SampleFormat::S24LE3 => for i in 0..self.len() {
                write_sample_s24le3(self.left[i], &mut buf);
                write_sample_s24le3(self.right[i], &mut buf);
            },
            SampleFormat::S24LE4 => for i in 0..self.len() {
                write_sample_s24le4(self.left[i], &mut buf);
                write_sample_s24le4(self.right[i], &mut buf);
            },
            SampleFormat::S32LE => for i in 0..self.len() {
                write_sample_s32le(self.left[i], &mut buf);
                write_sample_s32le(self.right[i], &mut buf);
            },
        }

        buf
    }

    pub fn from_buffer(
        format: SampleFormat,
        sample_rate: usize,
        buffer: &[u8],
        timestamp: Time,
    ) -> Frame {
        let samples = buffer.len() / format.bytes_per_sample() / CHANNELS;
        let mut frame = Frame::new(sample_rate, timestamp, samples);

        match format {
            SampleFormat::S16LE => for i in 0..samples {
                frame.left[i] = read_sample_s16le(&buffer, i * 4);
                frame.right[i] = read_sample_s16le(&buffer, i * 4 + 2);
            },
            SampleFormat::S24LE3 => for i in 0..samples {
                frame.left[i] = read_sample_s24le3(&buffer, i * 6);
                frame.right[i] = read_sample_s24le3(&buffer, i * 6 + 3);
            },
            SampleFormat::S24LE4 => for i in 0..samples {
                frame.left[i] = read_sample_s24le4(&buffer, i * 8);
                frame.right[i] = read_sample_s24le4(&buffer, i * 8 + 4);
            },
            SampleFormat::S32LE => for i in 0..samples {
                frame.left[i] = read_sample_s32le(&buffer, i * 8);
                frame.right[i] = read_sample_s32le(&buffer, i * 8 + 4);
            },
        }

        frame
    }
}

#[derive(Debug)]
pub struct Error {
    msg: String,
}

pub type Result<T> = result::Result<T, Error>;

impl Error {
    pub fn new(msg: &str) -> Error {
        Error {
            msg: String::from(msg),
        }
    }
    pub fn from_string(msg: String) -> Error {
        Error { msg: msg }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        return &self.msg;
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.msg)
    }
}

impl From<alsa::Error> for Error {
    fn from(e: alsa::Error) -> Error {
        Error::from_string(format!("Alsa error: {}", e))
    }
}


impl From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        Error::new(error::Error::description(&e))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct DeviceSpec {
    pub name: String,
    pub id: String,
    pub sample_rate: Option<usize>,
}

impl DeviceSpec {
    pub fn parse(spec_str: &str) -> Result<DeviceSpec> {
        let (name_id, rate) = match spec_str.rfind("#") {
            Some(p) => {
                let rate = match spec_str[p + 1..].parse::<usize>() {
                    Ok(r) if r > 0 && r < 200000 => r,
                    _ => {
                        return Err(Error::new(
                            &format!("Failed to parse sample rate: {}", spec_str),
                        ))
                    }
                };
                (&spec_str[..p], Some(rate))
            }
            None => (&spec_str[..], None),
        };

        let (name, id) = match name_id.find("@") {
            Some(p) => (&name_id[..p], &name_id[p + 1..]),
            None => (&name_id[..], &name_id[..]),
        };

        Ok(DeviceSpec {
            name: name.to_string(),
            id: id.to_string(),
            sample_rate: rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use base::*;

    #[test]
    fn frame_read_write_s16le() {
        // Verify that converting S16LE->Frame->S16LE doesn't change any data.
        let mut buf: Vec<u8> = vec![0u8; 0];
        for i in 0..32767 {
            buf.push((i & 0xff) as u8);
            buf.push(((i & 0xff00) >> 8) as u8);

            let r = (-(i as i32)) as u32;
            buf.push((r & 0xff) as u8);
            buf.push(((r & 0xff00) >> 8) as u8);
        }

        let frame = Frame::from_buffer(SampleFormat::S16LE, 44100, &buf[..]);
        let buf2 = frame.to_buffer(SampleFormat::S16LE);

        assert_eq!(buf.len(), buf2.len());
        for i in 0..buf.len() {
            assert_eq!(buf[i], buf2[i]);
        }
    }

    #[test]
    fn clamping() {
        let mut frame = Frame::new(100, 1);
        frame.left[0] = 1.5;
        frame.right[0] = -1.5;

        let buf16 = frame.to_buffer(SampleFormat::S16LE);

        // 32767
        assert_eq!(buf16[0], 0xff);
        assert_eq!(buf16[1], 0x7f);

        // -32768
        assert_eq!(buf16[2], 0x00);
        assert_eq!(buf16[3], 0x80);

        let buf24 = frame.to_buffer(SampleFormat::S24LE3);

        assert_eq!(buf24[0], 0xff);
        assert_eq!(buf24[1], 0xff);
        assert_eq!(buf24[2], 0x7f);

        assert_eq!(buf24[3], 0x00);
        assert_eq!(buf24[4], 0x00);
        assert_eq!(buf24[5], 0x80);

        let buf32 = frame.to_buffer(SampleFormat::S32LE);

        assert_eq!(buf32[0], 0xff);
        assert_eq!(buf32[1], 0xff);
        assert_eq!(buf32[2], 0xff);
        assert_eq!(buf32[3], 0x7f);

        assert_eq!(buf32[4], 0x00);
        assert_eq!(buf32[5], 0x00);
        assert_eq!(buf32[6], 0x00);
        assert_eq!(buf32[7], 0x80);
    }
}
