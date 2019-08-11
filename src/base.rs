#[cfg(feature = "packed_simd")]
use packed_simd;

use crate::time::{Time, TimeDelta};
use byteorder::{ByteOrder, LittleEndian};
use serde;
use std;
use std::cmp::{Eq, PartialEq};
use std::collections::VecDeque;
use std::error;
use std::fmt;
use std::io;
use std::iter::Enumerate;
use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::result;
use std::slice;

#[cfg(feature = "packed_simd")]
use self::packed_simd::f32x4;

// Subwoofer channel is expected to be reproduced 10dB louder
// than other channels.
pub const SUBWOOFER_LEVEL: f32 = 3.16227766017;

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

pub type ChannelPos = u8;

pub const CHANNEL_UNDEFINED: ChannelPos = 255;
pub const CHANNEL_FL: ChannelPos = 0;
pub const CHANNEL_FR: ChannelPos = 1;
pub const CHANNEL_FC: ChannelPos = 2;
pub const CHANNEL_SL: ChannelPos = 3;
pub const CHANNEL_SR: ChannelPos = 4;
pub const CHANNEL_SC: ChannelPos = 5;
pub const CHANNEL_LFE: ChannelPos = 7;
pub const CHANNEL_DYNAMIC_BASE: ChannelPos = 8;

pub const CHANNEL_MAX: ChannelPos = 20;

pub const NUM_CHANNEL_MAX: usize = CHANNEL_MAX as usize;

pub fn parse_channel_id(id: &str) -> Option<ChannelPos> {
    match id {
        "L" | "left" => Some(CHANNEL_FL),
        "R" | "right" => Some(CHANNEL_FR),
        "C" | "center" | "centre" => Some(CHANNEL_FC),
        "SL" | "surround_left" => Some(CHANNEL_SL),
        "SR" | "surround_right" => Some(CHANNEL_SR),
        "SC" | "surround" | "surround_center" | "surround_centre" => Some(CHANNEL_SC),
        "LFE" => Some(CHANNEL_LFE),
        "_" => Some(CHANNEL_UNDEFINED),
        _ => None,
    }
}

#[derive(Clone)]
pub struct PerChannel<T> {
    values: [Option<T>; NUM_CHANNEL_MAX],
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
            values: Default::default(),
        }
    }

    pub fn get(&self, c: ChannelPos) -> Option<&T> {
        assert!(c < CHANNEL_MAX);
        self.values[c as usize].as_ref()
    }

    pub fn get_mut(&mut self, c: ChannelPos) -> Option<&mut T> {
        assert!(c < CHANNEL_MAX);
        self.values[c as usize].as_mut()
    }

    pub fn take(&mut self, c: ChannelPos) -> Option<T> {
        assert!(c < CHANNEL_MAX);
        self.values[c as usize].take()
    }

    pub fn get_or_insert<F: FnOnce() -> T>(&mut self, c: ChannelPos, default: F) -> &mut T {
        assert!(c < CHANNEL_MAX);
        if !self.have_channel(c) {
            self.values[c as usize] = Some(default());
        }
        self.values[c as usize].as_mut().unwrap()
    }

    pub fn have_channel(&self, c: ChannelPos) -> bool {
        assert!(c < CHANNEL_MAX);
        self.values[c as usize].is_some()
    }

    pub fn set(&mut self, c: ChannelPos, v: T) {
        assert!(c < CHANNEL_MAX);
        self.values[c as usize] = Some(v)
    }

    pub fn iter(&mut self) -> ChannelIter<T> {
        ChannelIter::new(self)
    }

    pub fn clear(&mut self) {
        self.values = Default::default();
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
                        return v.as_mut().map(|v| (i as ChannelPos, v));
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
        + 127.5)
        / 2147483648f64) as f32
}

fn read_sample_s24le4(buf: &[u8]) -> f32 {
    (((((buf[0] as u32) << 8) | ((buf[1] as u32) << 16) | ((buf[2] as u32) << 24)) as i32 as f64
        + 127.5)
        / 2147483648f64) as f32
}

fn read_sample_s32le(buf: &[u8]) -> f32 {
    (((((buf[0] as u32) << 0)
        | ((buf[1] as u32) << 8)
        | ((buf[2] as u32) << 16)
        | ((buf[3] as u32) << 24)) as i32 as f64
        + 0.5)
        / 2147483648f64) as f32
}

// Fast SIMD-optimized convolution. Optimized for NEON on Raspberry PI 3.
#[cfg(feature = "packed_simd")]
pub fn convolve(v1: &[f32], v2: &[f32]) -> f32 {
    assert!(v1.len() == v2.len());

    let mut sum1 = f32x4::splat(0.0);
    let mut sum2 = f32x4::splat(0.0);

    for i in 0..(v1.len() / 8) {
        let v1_0 = f32x4::from_slice_unaligned(&v1[i * 8..]);
        let v1_4 = f32x4::from_slice_unaligned(&v1[i * 8 + 4..]);

        let v2_0 = f32x4::from_slice_unaligned(&v2[i * 8..]);
        let v2_4 = f32x4::from_slice_unaligned(&v2[i * 8 + 4..]);

        sum1 = sum1 + v1_0 * v2_0;
        sum2 = sum2 + v1_4 * v2_4;
    }

    let mut pos = (v1.len() / 8) * 8;
    while pos + 4 <= v1.len() {
        sum1 = sum1
            + f32x4::from_slice_unaligned(&v1[pos..]) * f32x4::from_slice_unaligned(&v2[pos..]);
        pos += 4;
    }

    let mut sum_end = 0.0;
    while pos < v1.len() {
        sum_end += v1[pos] * v2[pos];
        pos += 1;
    }

    sum1.extract(0)
        + sum1.extract(1)
        + sum1.extract(2)
        + sum1.extract(3)
        + sum2.extract(0)
        + sum2.extract(1)
        + sum2.extract(2)
        + sum2.extract(3)
        + sum_end
}

#[cfg(not(feature = "packed_simd"))]
pub fn convolve(v1: &[f32], v2: &[f32]) -> f32 {
    let mut r = 0.0;
    let mut block_count = v1.len() / 4;
    unsafe {
        if v1.len() == 0 {
            return 0.0;
        }
        let mut p1 = &v1[0] as *const f32;
        let mut p2 = &v2[0] as *const f32;
        while block_count > 0 {
            r += *p1 * *p2;
            p1 = p1.add(1);
            p2 = p2.add(1);
            r += *p1 * *p2;
            p1 = p1.add(1);
            p2 = p2.add(1);
            r += *p1 * *p2;
            p1 = p1.add(1);
            p2 = p2.add(1);
            r += *p1 * *p2;
            p1 = p1.add(1);
            p2 = p2.add(1);
            block_count -= 1;
        }

        block_count = v1.len() % 4;
        while block_count > 0 {
            r += *p1 * *p2;
            p1 = p1.add(1);
            p2 = p2.add(1);
            block_count -= 1;
        }
    }
    r
}

pub fn samples_to_timedelta(sample_rate: f64, samples: i64) -> TimeDelta {
    TimeDelta::seconds_f(samples as f64 / sample_rate)
}

pub fn get_sample_timestamp(start: Time, sample_rate: f64, sample: i64) -> Time {
    start + samples_to_timedelta(sample_rate, sample)
}

#[derive(Copy, Clone, Debug)]
pub struct Gain {
    pub db: f32,
}

impl Gain {
    pub fn zero() -> Gain {
        Gain { db: 0.0 }
    }
    pub fn get_multiplier(&self) -> f32 {
        10f32.powf(self.db / 20.0)
    }
    pub fn from_level(level: f32) -> Gain {
        Gain {
            db: level.log(10.0) * 20.0,
        }
    }
}

impl Add for Gain {
    type Output = Gain;
    fn add(self, other: Gain) -> Gain {
        Gain {
            db: self.db + other.db,
        }
    }
}

impl AddAssign for Gain {
    fn add_assign(&mut self, other: Gain) {
        self.db += other.db;
    }
}

impl Sub for Gain {
    type Output = Gain;
    fn sub(self, other: Gain) -> Gain {
        Gain {
            db: self.db - other.db,
        }
    }
}

impl SubAssign for Gain {
    fn sub_assign(&mut self, other: Gain) {
        self.db -= other.db;
    }
}

impl PartialEq for Gain {
    fn eq(&self, other: &Gain) -> bool {
        self.db == other.db
    }
}
impl Eq for Gain {}

impl serde::ser::Serialize for Gain {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_f32(self.db)
    }
}

pub struct Frame {
    pub sample_rate: f64,
    pub timestamp: Time,
    pub gain: Gain,
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
    pub fn new(sample_rate: f64, timestamp: Time, len: usize) -> Frame {
        Frame {
            sample_rate: sample_rate,
            timestamp: timestamp,
            gain: Gain::zero(),
            channels: PerChannel::new(),
            len: len,
            num_channels: 0,
        }
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
        if self.num_channels == 0 && self.len == 0 {
            self.len = samples.len();
        }

        assert!(samples.len() == self.len);
        if !self.channels.have_channel(pos) {
            self.num_channels += 1;
        }
        self.channels.set(pos, samples);
    }

    pub fn mix_channel(&mut self, pos: ChannelPos, samples: Vec<f32>, level: f32) {
        assert!(samples.len() == self.len);

        if !self.channels.have_channel(pos) && level == 1.0 {
            self.num_channels += 1;
            self.channels.set(pos, samples);
        } else {
            let data = self.ensure_channel(pos);
            for i in 0..data.len() {
                data[i] += samples[i] * level;
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
        samples_to_timedelta(self.sample_rate, self.len() as i64)
    }

    pub fn end_timestamp(&self) -> Time {
        self.timestamp + self.duration()
    }

    pub fn to_buffer_with_channel_map(
        &self,
        format: SampleFormat,
        out_channels: &[ChannelPos],
    ) -> Vec<u8> {
        let bytes_per_frame = format.bytes_per_sample() * out_channels.len();
        let multiplier = self.gain.get_multiplier();
        let mut buf = vec![0; bytes_per_frame * self.len()];
        for c in 0..out_channels.len() {
            if out_channels[c] == CHANNEL_UNDEFINED {
                continue;
            }

            assert!(out_channels[c] < CHANNEL_MAX);

            let data = match self.channels.get(out_channels[c]) {
                Some(data) => data,
                None => continue,
            };

            let shift = c * format.bytes_per_sample();
            match format {
                SampleFormat::S16LE => {
                    for i in 0..self.len() {
                        write_sample_s16le(
                            data[i] * multiplier,
                            &mut buf[i * bytes_per_frame + shift..],
                        );
                    }
                }
                SampleFormat::S24LE3 => {
                    for i in 0..self.len() {
                        write_sample_s24le3(
                            data[i] * multiplier,
                            &mut buf[i * bytes_per_frame + shift..],
                        );
                    }
                }
                SampleFormat::S24LE4 => {
                    for i in 0..self.len() {
                        write_sample_s24le4(
                            data[i] * multiplier,
                            &mut buf[i * bytes_per_frame + shift..],
                        );
                    }
                }
                SampleFormat::S32LE => {
                    for i in 0..self.len() {
                        write_sample_s32le(
                            data[i] * multiplier,
                            &mut buf[i * bytes_per_frame + shift..],
                        );
                    }
                }
                SampleFormat::F32LE => {
                    for i in 0..self.len() {
                        LittleEndian::write_f32(
                            &mut buf[i * bytes_per_frame + shift..],
                            data[i] * multiplier,
                        );
                    }
                }
            }
        }

        buf
    }

    pub fn from_buffer_stereo(
        format: SampleFormat,
        sample_rate: f64,
        buffer: &[u8],
        timestamp: Time,
    ) -> Frame {
        Frame::from_buffer(
            format,
            sample_rate,
            &[CHANNEL_FL, CHANNEL_FR],
            buffer,
            timestamp,
        )
    }

    pub fn from_buffer(
        format: SampleFormat,
        sample_rate: f64,
        channels: &[ChannelPos],
        buffer: &[u8],
        timestamp: Time,
    ) -> Frame {
        let samples = buffer.len() / format.bytes_per_sample() / channels.len();
        let mut frame = Frame::new(sample_rate, timestamp, samples);

        let bytes_per_sample = format.bytes_per_sample() * channels.len();

        for c in 0..channels.len() {
            let data = frame.ensure_channel(channels[c]);
            match format {
                SampleFormat::S16LE => {
                    for i in 0..samples {
                        data[i] = read_sample_s16le(&buffer[i * bytes_per_sample + c * 2..]);
                    }
                }
                SampleFormat::S24LE3 => {
                    for i in 0..samples {
                        data[i] = read_sample_s24le3(&buffer[i * bytes_per_sample + c * 3..]);
                    }
                }
                SampleFormat::S24LE4 => {
                    for i in 0..samples {
                        data[i] = read_sample_s24le4(&buffer[i * bytes_per_sample + c * 4..]);
                    }
                }
                SampleFormat::S32LE => {
                    for i in 0..samples {
                        data[i] = read_sample_s32le(&buffer[i * bytes_per_sample + c * 4..]);
                    }
                }
                SampleFormat::F32LE => {
                    for i in 0..samples {
                        data[i] = LittleEndian::read_f32(&buffer[i * bytes_per_sample + c * 4..])
                            + 1e-10f32;
                    }
                }
            }
        }

        frame
    }

    pub fn have_channel(&self, pos: ChannelPos) -> bool {
        self.channels.have_channel(pos)
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

impl From<rppal::gpio::Error> for Error {
    fn from(e: rppal::gpio::Error) -> Error {
        Error::from_string(format!("GPIO error: {}", e))
    }
}

impl<R: pest::RuleType> From<pest::error::Error<R>> for Error {
    fn from(e: pest::error::Error<R>) -> Error {
        Error::from_string(format!("Failed to parse format expression: {}", e))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct DeviceSpec {
    pub name: String,
    pub id: String,
    pub sample_rate: Option<usize>,
    pub channels: Vec<ChannelPos>,
    pub delay: TimeDelta,
    pub enable_a52: bool,
}

pub struct SeriesStats {
    window: usize,
    values: VecDeque<f64>,
    sum: f64,
}

impl SeriesStats {
    pub fn new(window: usize) -> SeriesStats {
        return SeriesStats {
            window: window,
            values: VecDeque::new(),
            sum: 0.0f64,
        };
    }

    pub fn average(&self) -> Option<f64> {
        if self.values.is_empty() {
            None
        } else {
            Some(self.sum / self.values.len() as f64)
        }
    }

    pub fn push(&mut self, v: f64) {
        self.values.push_back(v);
        self.sum += v;
        if self.values.len() > self.window {
            self.sum -= self.values.pop_front().unwrap();
        }
    }

    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }
}

pub struct StreamPositionTracker {
    base_time: Time,
    base_sample_rate: f64,
    sample_rate: f64,
    samples_pos: f64,
    offset: SeriesStats,

    prev_clock_drift: Option<TimeDelta>,
    clock_drift: Option<TimeDelta>,
}

const RATE_UPDATE_PERIOD_S: f64 = 2.0;

// How much measured sample rate is allowed to deviate from the expected value.
// For 48kHz 0.3% is 144, i.e. sample rate can change between 47856 and 48144.
const MAX_RATE_DEVIATION: f64 = 0.003;

impl StreamPositionTracker {
    pub fn new(sample_rate: f64) -> StreamPositionTracker {
        StreamPositionTracker {
            base_time: Time::zero(),
            base_sample_rate: sample_rate,
            sample_rate: sample_rate,
            samples_pos: 0.0,
            offset: SeriesStats::new(200),
            prev_clock_drift: None,
            clock_drift: None,
        }
    }

    pub fn offset(&self) -> f64 {
        self.offset.average().unwrap_or(0.0)
    }

    pub fn pos_no_offset(&self) -> Time {
        let pos_s = self.samples_pos / self.sample_rate;
        self.base_time + TimeDelta::seconds_f(pos_s)
    }

    pub fn pos(&self) -> Time {
        let pos_s = self.samples_pos / self.sample_rate + self.offset.average().unwrap_or(0.0);
        self.base_time + TimeDelta::seconds_f(pos_s)
    }

    pub fn set_target_pos(&mut self, target_pos: Option<Time>) {
        self.clock_drift = target_pos.map(|t| t - self.pos());
    }

    pub fn add_samples(&mut self, samples: usize, pos_estimate: Time) {
        if self.base_time == Time::zero() {
            self.base_time = pos_estimate;
            self.samples_pos = 0.0;
        } else {
            self.samples_pos += samples as f64;
            let new_offset = (pos_estimate - self.base_time).in_seconds_f()
                - self.samples_pos / self.sample_rate;
            self.offset.push(new_offset);
        }
    }

    pub fn reset(&mut self, base_time: Time, base_sample_rate: f64) {
        self.base_time = base_time;
        self.base_sample_rate = base_sample_rate;
        self.sample_rate = self.base_sample_rate;
        self.samples_pos = 0.0;
        self.offset.reset();
        self.clock_drift = None;
        self.prev_clock_drift = None;
    }

    pub fn update_sample_rate(&mut self) -> Option<f64> {
        if self.samples_pos < RATE_UPDATE_PERIOD_S * self.sample_rate {
            return None;
        }

        let diff = match (self.clock_drift, self.prev_clock_drift) {
            (Some(clock_drift), Some(prev_clock_drift)) => {
                (clock_drift - prev_clock_drift + clock_drift / 5).in_seconds_f()
            }
            _ => 0.0,
        };
        let mut new_sample_rate =
            self.sample_rate + diff * self.sample_rate * self.sample_rate / self.samples_pos / 2.0;

        let min = self.base_sample_rate * (1.0 - MAX_RATE_DEVIATION);
        if new_sample_rate < min {
            new_sample_rate = min;
        }
        let max = self.base_sample_rate * (1.0 + MAX_RATE_DEVIATION);
        if new_sample_rate > max {
            new_sample_rate = max;
        }

        println!(
            "offset {} diff {} new_rate {}.",
            self.offset() * 1000.0,
            diff * 1000.0,
            new_sample_rate
        );

        self.prev_clock_drift = self.clock_drift;
        self.base_time = self.pos_no_offset();
        self.samples_pos = 0.0;
        self.sample_rate = new_sample_rate;

        Some(new_sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let buf2 = frame.to_buffer_with_channel_map(SampleFormat::S16LE, &[CHANNEL_FL, CHANNEL_FR]);

        assert_eq!(buf.len(), buf2.len());
        for i in 0..buf.len() {
            assert_eq!(buf[i], buf2[i]);
        }
    }

    #[test]
    fn clamping() {
        let mut frame = Frame::new(100.0, Time::now(), 1);
        frame.ensure_channel(CHANNEL_FL)[0] = 1.5;
        frame.ensure_channel(CHANNEL_FR)[0] = -1.5;

        let buf16 =
            frame.to_buffer_with_channel_map(SampleFormat::S16LE, &[CHANNEL_FL, CHANNEL_FR]);

        // 32767
        assert_eq!(buf16[0], 0xff);
        assert_eq!(buf16[1], 0x7f);

        // -32768
        assert_eq!(buf16[2], 0x00);
        assert_eq!(buf16[3], 0x80);

        let buf24 =
            frame.to_buffer_with_channel_map(SampleFormat::S24LE3, &[CHANNEL_FL, CHANNEL_FR]);

        assert_eq!(buf24[0], 0xff);
        assert_eq!(buf24[1], 0xff);
        assert_eq!(buf24[2], 0x7f);

        assert_eq!(buf24[3], 0x00);
        assert_eq!(buf24[4], 0x00);
        assert_eq!(buf24[5], 0x80);

        let buf32 =
            frame.to_buffer_with_channel_map(SampleFormat::S32LE, &[CHANNEL_FL, CHANNEL_FR]);

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
