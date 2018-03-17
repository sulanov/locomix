extern crate alsa;
extern crate byteorder;
extern crate simd;

use self::byteorder::{ByteOrder, LittleEndian};
use self::simd::f32x4;
use std::error;
use std::fmt;
use std::io;
use std::slice;
use std::iter::Enumerate;
use std::result;
use std::collections::VecDeque;
use time::{Time, TimeDelta};

#[derive(Clone, Copy, Debug)]
pub enum SampleFormat {
    S16LE,
    S24LE3,
    S24LE4,
    S32LE,
    F32LE,
}

impl SampleFormat {
    pub fn to_str(&self) -> &'static str {
        match *self {
            SampleFormat::S16LE => "S16LE",
            SampleFormat::S24LE3 => "S24LE3",
            SampleFormat::S24LE4 => "S24LE4",
            SampleFormat::S32LE => "S32LE",
            SampleFormat::F32LE => "F32LE",
        }
    }

    pub fn bytes_per_sample(&self) -> usize {
        match *self {
            SampleFormat::S16LE => 2,
            SampleFormat::S24LE3 => 3,
            SampleFormat::S24LE4 => 4,
            SampleFormat::S32LE => 4,
            SampleFormat::F32LE => 4,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Ord, PartialOrd)]
pub enum ChannelPos {
    Other = -1,
    FL = 0,
    FR = 1,
    FC = 2,
    SL = 3,
    SR = 4,
    SC = 5,
    Sub = 6,
}

const CHANNEL_MAX: usize = 7;
const ALL_CHANNEL: [ChannelPos; CHANNEL_MAX] = [
    ChannelPos::FL,
    ChannelPos::FR,
    ChannelPos::FC,
    ChannelPos::SL,
    ChannelPos::SR,
    ChannelPos::SC,
    ChannelPos::Sub,
];

#[derive(Clone)]
pub struct PerChannel<T> {
    values: [Option<T>; CHANNEL_MAX],
}

pub struct ChannelIter<'a, T: 'a> {
    inner: Enumerate<slice::IterMut<'a, Option<T>>>,
}

impl<'a, T> ChannelIter<'a, T> {
    fn new(per_channel: &'a mut PerChannel<T>) -> ChannelIter<T> {
        ChannelIter {
            inner: per_channel.values.iter_mut().enumerate(),
        }
    }
}

impl<T> PerChannel<T> {
    pub fn new() -> PerChannel<T> {
        PerChannel {
            values: [None, None, None, None, None, None, None],
        }
    }

    pub fn get(&self, c: ChannelPos) -> Option<&T> {
        assert!(c != ChannelPos::Other);
        self.values[c as usize].as_ref()
    }

    pub fn get_mut(&mut self, c: ChannelPos) -> Option<&mut T> {
        assert!(c != ChannelPos::Other);
        self.values[c as usize].as_mut()
    }

    pub fn take(&mut self, c: ChannelPos) -> Option<T> {
        assert!(c != ChannelPos::Other);
        self.values[c as usize].take()
    }

    pub fn get_or_insert<F: FnOnce() -> T>(&mut self, c: ChannelPos, default: F) -> &mut T {
        assert!(c != ChannelPos::Other);
        if !self.have_channel(c) {
            self.values[c as usize] = Some(default());
        }
        self.values[c as usize].as_mut().unwrap()
    }

    pub fn have_channel(&self, c: ChannelPos) -> bool {
        assert!(c != ChannelPos::Other);
        self.values[c as usize].is_some()
    }

    pub fn set(&mut self, c: ChannelPos, v: T) {
        assert!(c != ChannelPos::Other);
        self.values[c as usize] = Some(v)
    }

    pub fn iter(&mut self) -> ChannelIter<T> {
        ChannelIter::new(self)
    }

    pub fn clear(&mut self) {
        self.values = [None, None, None, None, None, None, None];
    }
}

impl<'a, T> Iterator for ChannelIter<'a, T> {
    type Item = (ChannelPos, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                None => return None,
                Some((i, v)) => {
                    if v.is_none() {
                        continue;
                    } else {
                        return v.as_mut().map(|v| (ALL_CHANNEL[i], v));
                    }
                }
            }
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

pub fn samples_to_timedelta(sample_rate: f32, samples: i64) -> TimeDelta {
    (TimeDelta::seconds(1) * samples) / sample_rate as i64
}

pub fn get_sample_timestamp(start: Time, sample_rate: f32, sample: i64) -> Time {
    start + samples_to_timedelta(sample_rate, sample)
}

pub struct Frame {
    pub sample_rate: f32,
    pub timestamp: Time,
    channels: PerChannel<Vec<f32>>,
    len: usize,
    num_channels: usize,
}

pub struct FrameChannelIter<'a> {
    iter: ChannelIter<'a, Vec<f32>>,
}

impl<'a> Iterator for FrameChannelIter<'a> {
    type Item = (ChannelPos, &'a mut [f32]);

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            None => None,
            Some((c, v)) => Some((c, &mut v[..])),
        }
    }
}

impl Frame {
    pub fn new(sample_rate: f32, timestamp: Time, len: usize) -> Frame {
        Frame {
            sample_rate: sample_rate,
            timestamp: timestamp,
            channels: PerChannel::new(),
            len: len,
            num_channels: 0,
        }
    }

    pub fn new_stereo(sample_rate: f32, timestamp: Time, len: usize) -> Frame {
        let mut result = Frame::new(sample_rate, timestamp, len);
        result.ensure_channel(ChannelPos::FL);
        result.ensure_channel(ChannelPos::FR);
        result
    }

    pub fn get_channel(&self, pos: ChannelPos) -> Option<&[f32]> {
        self.channels.get(pos).map(|c| &c[..])
    }

    pub fn get_channel_mut(&mut self, pos: ChannelPos) -> Option<&mut [f32]> {
        self.channels.get_mut(pos).map(|c| &mut c[..])
    }

    pub fn ensure_channel(&mut self, pos: ChannelPos) -> &mut [f32] {
        let mut added = false;
        let len = self.len;
        let result = &mut (self.channels.get_or_insert(pos, || {
            added = true;
            // Avoid denormal zero.
            vec![1e-10f32; len]
        })[..]);
        if added {
            self.num_channels += 1;
        }
        result
    }

    pub fn set_channel(&mut self, pos: ChannelPos, samples: Vec<f32>) {
        assert!(samples.len() == self.len);
        if !self.channels.have_channel(pos) {
            self.num_channels += 1;
        }
        self.channels.set(pos, samples);
    }

    pub fn mix_channel(&mut self, pos: ChannelPos, samples: Vec<f32>) {
        assert!(samples.len() == self.len);

        if !self.channels.have_channel(pos) {
            self.num_channels += 1;
            self.channels.set(pos, samples);
        } else {
            let data = self.channels.get_mut(pos).unwrap();
            for i in 0..data.len() {
                data[i] += samples[i];
            }
        }
    }

    pub fn take_channel(&mut self, pos: ChannelPos) -> Option<Vec<f32>> {
        if self.channels.have_channel(pos) {
            self.num_channels -= 1;
        }
        self.channels.take(pos)
    }

    pub fn take_channel_or_zero(&mut self, pos: ChannelPos) -> Vec<f32> {
        self.channels
            .take(pos)
            .unwrap_or_else(|| vec![1e-10f32; self.len])
    }

    pub fn iter_channels(&mut self) -> FrameChannelIter {
        FrameChannelIter {
            iter: self.channels.iter(),
        }
    }

    pub fn channels(&self) -> usize {
        self.num_channels
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn duration(&self) -> TimeDelta {
        (TimeDelta::seconds(1) * self.len() as i64) / self.sample_rate as i64
    }

    pub fn end_timestamp(&self) -> Time {
        self.timestamp + self.duration()
    }

    // pub fn to_buffer(&self, format: SampleFormat) -> Vec<u8> {
    //     let chmap: Vec<ChannelPos> = self.channels.iter().map(|c| c.pos).collect();
    //     self.to_buffer_with_channel_map(format, chmap.as_slice())
    // }

    pub fn to_buffer_with_channel_map(
        &self,
        format: SampleFormat,
        out_channels: &[ChannelPos],
    ) -> Vec<u8> {
        let bytes_per_frame = format.bytes_per_sample() * out_channels.len();
        let mut buf = vec![0; bytes_per_frame * self.len()];
        for c in 0..out_channels.len() {
            if out_channels[c] == ChannelPos::Other {
                continue;
            }
            let data = match self.channels.get(out_channels[c]) {
                Some(data) => data,
                None => continue,
            };

            let shift = c * format.bytes_per_sample();
            match format {
                SampleFormat::S16LE => for i in 0..self.len() {
                    write_sample_s16le(data[i], &mut buf[i * bytes_per_frame + shift..]);
                },
                SampleFormat::S24LE3 => for i in 0..self.len() {
                    write_sample_s24le3(data[i], &mut buf[i * bytes_per_frame + shift..]);
                },
                SampleFormat::S24LE4 => for i in 0..self.len() {
                    write_sample_s24le4(data[i], &mut buf[i * bytes_per_frame + shift..]);
                },
                SampleFormat::S32LE => for i in 0..self.len() {
                    write_sample_s32le(data[i], &mut buf[i * bytes_per_frame + shift..]);
                },
                SampleFormat::F32LE => for i in 0..self.len() {
                    LittleEndian::write_f32(&mut buf[i * bytes_per_frame + shift..], data[i]);
                },
            }
        }

        buf
    }

    pub fn from_buffer_stereo(
        format: SampleFormat,
        sample_rate: f32,
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
        sample_rate: f32,
        channels: &[ChannelPos],
        buffer: &[u8],
        timestamp: Time,
    ) -> Frame {
        let samples = buffer.len() / format.bytes_per_sample() / channels.len();
        let mut frame = Frame::new(sample_rate, timestamp, samples);

        let bytes_per_sample = format.bytes_per_sample() * channels.len();

        for c in 0..channels.len() {
            let mut data = frame.ensure_channel(channels[c]);
            match format {
                SampleFormat::S16LE => for i in 0..samples {
                    data[i] = read_sample_s16le(&buffer[i * bytes_per_sample + c * 2..]);
                },
                SampleFormat::S24LE3 => for i in 0..samples {
                    data[i] = read_sample_s24le3(&buffer[i * bytes_per_sample + c * 3..]);
                },
                SampleFormat::S24LE4 => for i in 0..samples {
                    data[i] = read_sample_s24le4(&buffer[i * bytes_per_sample + c * 4..]);
                },
                SampleFormat::S32LE => for i in 0..samples {
                    data[i] = read_sample_s32le(&buffer[i * bytes_per_sample + c * 4..]);
                },
                SampleFormat::F32LE => for i in 0..samples {
                    data[i] = LittleEndian::read_f32(&buffer[i * bytes_per_sample + c * 4..]);
                },
            }
        }

        frame
    }

    pub fn have_channel(&self, pos: ChannelPos) -> bool {
        self.channels.have_channel(pos)
    }

    pub fn is_stereo(&self) -> bool {
        self.channels() == 2 && self.have_channel(ChannelPos::FL)
            && self.have_channel(ChannelPos::FR)
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
    pub exact_sample_rate: bool,
    pub channels: Vec<ChannelPos>,
    pub delay: TimeDelta,
    pub enable_a52: bool,
}

const TIME_PRECISION_US: i64 = 1000;
const MAX_SAMPLE_RATE: f32 = 96000.0;

pub struct RateDetector {
    window: TimeDelta,
    history: VecDeque<(Time, usize)>,
    sum: usize,
}

#[derive(Clone, Copy)]
pub struct DetectedRate {
    pub rate: f32,
    pub error: f32,
}

impl RateDetector {
    pub fn new(target_precision: f32) -> RateDetector {
        return RateDetector {
            window: TimeDelta::microseconds(TIME_PRECISION_US)
                * (2.0 * MAX_SAMPLE_RATE / target_precision),
            history: VecDeque::new(),
            sum: 0,
        };
    }

    pub fn update(&mut self, samples: usize, t_end: Time) -> DetectedRate {
        self.sum += samples;
        self.history.push_back((t_end, samples));
        while self.history.len() > 0 && t_end - self.history[0].0 >= self.window {
            self.sum -= self.history.pop_front().unwrap().1;
        }

        let period = t_end - self.history[0].0;
        if period <= TimeDelta::zero() {
            return DetectedRate {
                rate: MAX_SAMPLE_RATE / 2.0,
                error: MAX_SAMPLE_RATE / 2.0,
            };
        }

        let samples = (self.sum - self.history[0].1) as f64;
        let rate = samples / period.in_seconds_f();
        DetectedRate {
            rate: rate as f32,
            error: (rate * (TIME_PRECISION_US as f64 / 1_000_000.0) / period.in_seconds_f()) as f32,
        }
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.sum = 0;
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

        let frame = Frame::from_buffer_stereo(SampleFormat::S16LE, 44100.0, &buf[..], Time::now());
        let buf2 = frame.to_buffer(SampleFormat::S16LE);

        assert_eq!(buf.len(), buf2.len());
        for i in 0..buf.len() {
            assert_eq!(buf[i], buf2[i]);
        }
    }

    #[test]
    fn clamping() {
        let mut frame = Frame::new_stereo(100.0, Time::now(), 1);
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
