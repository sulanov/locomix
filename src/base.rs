extern crate alsa;
extern crate simd;

use self::simd::f32x4;
use std::error;
use std::fmt;
use std::io;
use std::result;
use time::{Time, TimeDelta};

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Ord, PartialOrd)]
pub enum ChannelPos {
    FL,
    FR,
    Sub,
    Other,
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

fn write_sample_s16le(val: f32, buf: &mut [u8]) {
    let ival = (clamp(val) as f64 * 2147483648f64 - 32767.5) as i32 as u32;
    buf[0] = ((ival & 0x00ff0000) >> 16) as u8;
    buf[1] = ((ival & 0xff000000) >> 24) as u8;
}

fn write_sample_s24le3(val: f32, buf: &mut [u8]) {
    let ival = (clamp(val) as f64 * 2147483648f64 - 127.5) as i32 as u32;
    buf[0] = ((ival & 0x0000ff00) >> 8) as u8;
    buf[1] = ((ival & 0x00ff0000) >> 16) as u8;
    buf[2] = ((ival & 0xff000000) >> 24) as u8;
}

fn write_sample_s24le4(val: f32, buf: &mut [u8]) {
    let ival = (clamp(val) as f64 * 2147483648f64 - 127.5) as i32 as u32;
    buf[0] = ((ival & 0x0000ff00) >> 8) as u8;
    buf[1] = ((ival & 0x00ff0000) >> 16) as u8;
    buf[2] = ((ival & 0xff000000) >> 24) as u8;
    buf[3] = 0;
}

fn write_sample_s32le(val: f32, buf: &mut [u8]) {
    let ival = (clamp(val) as f64 * 2147483648f64 - 0.5) as i32 as u32;
    buf[0] = ((ival & 0x000000ff) >> 0) as u8;
    buf[1] = ((ival & 0x0000ff00) >> 8) as u8;
    buf[2] = ((ival & 0x00ff0000) >> 16) as u8;
    buf[3] = ((ival & 0xff000000) >> 24) as u8;
}

fn read_sample_s16le(buf: &[u8]) -> f32 {
    (((((buf[0] as u32) << 16) | ((buf[1] as u32) << 24)) as i32 as f64 + 32767.5) / 2147483648f64)
        as f32
}

fn read_sample_s24le3(buf: &[u8]) -> f32 {
    (((((buf[0] as u32) << 8) | ((buf[1] as u32) << 16) | ((buf[2] as u32) << 24)) as i32 as f64
        + 127.5) / 2147483648f64) as f32
}

fn read_sample_s24le4(buf: &[u8]) -> f32 {
    (((((buf[0] as u32) << 8) | ((buf[1] as u32) << 16) | ((buf[2] as u32) << 24)) as i32 as f64
        + 127.5) / 2147483648f64) as f32
}

fn read_sample_s32le(buf: &[u8]) -> f32 {
    (((((buf[0] as u32) << 0) | ((buf[1] as u32) << 8) | ((buf[2] as u32) << 16)
        | ((buf[3] as u32) << 24)) as i32 as f64 + 0.5) / 2147483648f64) as f32
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

    sum1.extract(0) + sum1.extract(1) + sum1.extract(2) + sum1.extract(3) + sum2.extract(0)
        + sum2.extract(1) + sum2.extract(2) + sum2.extract(3) + sum_end
}

pub fn samples_to_timedelta(sample_rate: usize, samples: i64) -> TimeDelta {
    (TimeDelta::seconds(1) * samples) / sample_rate as i64
}

pub fn get_sample_timestamp(start: Time, sample_rate: usize, sample: i64) -> Time {
    start + samples_to_timedelta(sample_rate, sample)
}

pub struct ChannelData {
    pub pos: ChannelPos,
    pub pcm: Vec<f32>,
}

impl ChannelData {
    pub fn new(pos: ChannelPos, samples: usize) -> ChannelData {
        // Avoid denormal zero.
        let zero = 1e-10f32;
        ChannelData {
            pos: pos,
            pcm: vec![zero; samples],
        }
    }
}

pub struct Frame {
    pub sample_rate: usize,
    pub timestamp: Time,
    pub channels: Vec<ChannelData>,
}

impl Frame {
    pub fn new(sample_rate: usize, timestamp: Time) -> Frame {
        Frame {
            sample_rate: sample_rate,
            timestamp: timestamp,
            channels: Vec::new(),
        }
    }

    pub fn new_stereo(sample_rate: usize, timestamp: Time, samples: usize) -> Frame {
        Frame {
            sample_rate: sample_rate,
            timestamp: timestamp,
            channels: vec![
                ChannelData::new(ChannelPos::FL, samples),
                ChannelData::new(ChannelPos::FR, samples),
            ],
        }
    }

    pub fn len(&self) -> usize {
        if self.channels.is_empty() {
            0
        } else {
            self.channels[0].pcm.len()
        }
    }

    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    pub fn duration(&self) -> TimeDelta {
        (TimeDelta::seconds(1) * self.len() as i64) / self.sample_rate as i64
    }

    pub fn end_timestamp(&self) -> Time {
        self.timestamp + self.duration()
    }

    pub fn to_buffer(&self, format: SampleFormat) -> Vec<u8> {
        let chmap: Vec<ChannelPos> = self.channels.iter().map(|c| c.pos).collect();
        self.to_buffer_with_channel_map(format, chmap.as_slice())
    }

    pub fn to_buffer_with_channel_map(
        &self,
        format: SampleFormat,
        out_channels: &[ChannelPos],
    ) -> Vec<u8> {
        let bytes_per_frame = format.bytes_per_sample() * out_channels.len();
        let mut buf = vec![0; bytes_per_frame * self.len()];
        for channel in &self.channels {
            if channel.pos == ChannelPos::Other {
                continue;
            }

            let out_channel = match out_channels.iter().position(|pos| channel.pos == *pos) {
                Some(p) => p,
                None => continue,
            };

            let shift = out_channel * format.bytes_per_sample();
            match format {
                SampleFormat::S16LE => for i in 0..self.len() {
                    write_sample_s16le(channel.pcm[i], &mut buf[i * bytes_per_frame + shift..]);
                },
                SampleFormat::S24LE3 => for i in 0..self.len() {
                    write_sample_s24le3(channel.pcm[i], &mut buf[i * bytes_per_frame + shift..]);
                },
                SampleFormat::S24LE4 => for i in 0..self.len() {
                    write_sample_s24le4(channel.pcm[i], &mut buf[i * bytes_per_frame + shift..]);
                },
                SampleFormat::S32LE => for i in 0..self.len() {
                    write_sample_s32le(channel.pcm[i], &mut buf[i * bytes_per_frame + shift..]);
                },
            }
        }

        buf
    }

    pub fn from_buffer_stereo(
        format: SampleFormat,
        sample_rate: usize,
        buffer: &[u8],
        timestamp: Time,
    ) -> Frame {
        Frame::from_buffer(
            format,
            sample_rate,
            &[ChannelPos::FL, ChannelPos::FR],
            buffer,
            timestamp,
        )
    }

    pub fn from_buffer(
        format: SampleFormat,
        sample_rate: usize,
        channels: &[ChannelPos],
        buffer: &[u8],
        timestamp: Time,
    ) -> Frame {
        let samples = buffer.len() / format.bytes_per_sample() / channels.len();
        let mut frame = Frame::new(sample_rate, timestamp);

        let bytes_per_sample = format.bytes_per_sample() * channels.len();

        for c in 0..channels.len() {
            let mut data = ChannelData::new(channels[c], samples);
            match format {
                SampleFormat::S16LE => for i in 0..samples {
                    data.pcm[i] = read_sample_s16le(&buffer[i * bytes_per_sample + c * 2..]);
                },
                SampleFormat::S24LE3 => for i in 0..samples {
                    data.pcm[i] = read_sample_s24le3(&buffer[i * bytes_per_sample + c * 3..]);
                },
                SampleFormat::S24LE4 => for i in 0..samples {
                    data.pcm[i] = read_sample_s24le4(&buffer[i * bytes_per_sample + c * 4..]);
                },
                SampleFormat::S32LE => for i in 0..samples {
                    data.pcm[i] = read_sample_s32le(&buffer[i * bytes_per_sample + c * 4..]);
                },
            }
            frame.channels.push(data)
        }

        frame
    }

    pub fn is_stereo(&self) -> bool {
        self.channels.len() == 2 && self.channels[0].pos == ChannelPos::FL
            && self.channels[1].pos == ChannelPos::FR
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
    pub channels: Vec<ChannelPos>,
    pub delay: TimeDelta,
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

        let frame = Frame::from_buffer_stereo(SampleFormat::S16LE, 44100, &buf[..], Time::now());
        let buf2 = frame.to_buffer(SampleFormat::S16LE);

        assert_eq!(buf.len(), buf2.len());
        for i in 0..buf.len() {
            assert_eq!(buf[i], buf2[i]);
        }
    }

    #[test]
    fn clamping() {
        let mut frame = Frame::new_stereo(100, Time::now(), 1);
        frame.channels[0].pcm[0] = 1.5;
        frame.channels[1].pcm[0] = -1.5;

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
