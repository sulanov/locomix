#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::base;
use crate::time;
use libc::{c_int, c_void, uint32_t, uint8_t};
use std::cmp;
use std::slice;

const A52_MONO: c_int = 1;
const A52_STEREO: c_int = 2;
const A52_3F: c_int = 3;
const A52_2F1R: c_int = 4;
const A52_3F1R: c_int = 5;
const A52_2F2R: c_int = 6;
const A52_3F2R: c_int = 7;
const A52_LFE: c_int = 16;

const A52_MONO_LFE: c_int = A52_MONO | A52_LFE;
const A52_STEREO_LFE: c_int = A52_STEREO | A52_LFE;
const A52_3F_LFE: c_int = A52_3F | A52_LFE;
const A52_2F1R_LFE: c_int = A52_2F1R | A52_LFE;
const A52_3F1R_LFE: c_int = A52_3F1R | A52_LFE;
const A52_2F2R_LFE: c_int = A52_2F2R | A52_LFE;
const A52_3F2R_LFE: c_int = A52_3F2R | A52_LFE;

type sample_t = f32;

enum a52_state_t {}

type dynrng_callback = Option<extern "C" fn(sample_t, *mut c_void)>;

#[link(name = "a52")]
extern "C" {
    fn a52_init(mm_accel: uint32_t) -> *mut a52_state_t;
    fn a52_samples(state: *mut a52_state_t) -> *const sample_t;
    fn a52_syncinfo(
        buf: *const uint8_t,
        flags: *mut c_int,
        sample_rate: *mut c_int,
        bit_rate: *mut c_int,
    ) -> c_int;
    fn a52_frame(
        state: *mut a52_state_t,
        buf: *const uint8_t,
        flags: *mut c_int,
        level: *const sample_t,
        bias: sample_t,
    ) -> c_int;
    fn a52_dynrng(state: *mut a52_state_t, callback: dynrng_callback, data: *mut c_void);
    fn a52_block(state: *mut a52_state_t) -> c_int;
    fn a52_free(state: *mut a52_state_t);
}

const SPDIF_HEADER_SIZE: usize = 8;
const AC52_HEADER_SIZE: usize = 8;
const HEADER_SIZE: usize = SPDIF_HEADER_SIZE + AC52_HEADER_SIZE;

const SPDIF_SYNC_SIZE: usize = 4;
const SPDIF_SYNC: [u8; SPDIF_SYNC_SIZE] = [0xf8, 0x72, 0x4e, 0x1f];
const SPDIF_NULL: u8 = 0;
const SPDIF_AC3: u8 = 1;
const SPDIF_PAUSE: u8 = 3;

type ChannelMap = &'static [base::ChannelPos];

static CHMAP_MONO: ChannelMap = &[base::ChannelPos::FC];
static CHMAP_MONO_LFE: ChannelMap = &[base::ChannelPos::Sub, base::ChannelPos::FC];

static CHMAP_STEREO: ChannelMap = &[base::ChannelPos::FL, base::ChannelPos::FR];
static CHMAP_STEREO_LFE: ChannelMap = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FR,
];

static CHMAP_3F: ChannelMap = &[
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
];
static CHMAP_3F_LFE: ChannelMap = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
];

static CHMAP_2F1R: ChannelMap = &[
    base::ChannelPos::FL,
    base::ChannelPos::FR,
    base::ChannelPos::SC,
];
static CHMAP_2F1R_LFE: ChannelMap = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FR,
    base::ChannelPos::SC,
];
static CHMAP_3F1R: ChannelMap = &[
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
    base::ChannelPos::SC,
];
static CHMAP_3F1R_LFE: ChannelMap = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
    base::ChannelPos::SC,
];
static CHMAP_2F2R: ChannelMap = &[
    base::ChannelPos::FL,
    base::ChannelPos::FR,
    base::ChannelPos::SL,
    base::ChannelPos::SR,
];
static CHMAP_2F2R_LFE: ChannelMap = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FR,
    base::ChannelPos::SL,
    base::ChannelPos::SR,
];
static CHMAP_3F2R: ChannelMap = &[
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
    base::ChannelPos::SL,
    base::ChannelPos::SR,
];
static CHMAP_3F2R_LFE: ChannelMap = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
    base::ChannelPos::SL,
    base::ChannelPos::SR,
];

// Each frame contains 6 parts, 256 samples for each frame.
const PARTS_PER_FRAME: usize = 6;
const SAMPLES_PER_PART: usize = 256;
const SAMPLES_PER_FRAME: usize = PARTS_PER_FRAME * SAMPLES_PER_PART;

fn get_channel_map(flags: c_int) -> Option<&'static [base::ChannelPos]> {
    match flags & 0x17 {
        A52_MONO => Some(CHMAP_MONO),
        A52_MONO_LFE => Some(CHMAP_MONO_LFE),
        A52_STEREO => Some(CHMAP_STEREO),
        A52_STEREO_LFE => Some(CHMAP_STEREO_LFE),
        A52_3F => Some(CHMAP_3F),
        A52_3F_LFE => Some(CHMAP_3F_LFE),
        A52_2F1R => Some(CHMAP_2F1R),
        A52_2F1R_LFE => Some(CHMAP_2F1R_LFE),
        A52_3F1R => Some(CHMAP_3F1R),
        A52_3F1R_LFE => Some(CHMAP_3F1R_LFE),
        A52_2F2R => Some(CHMAP_2F2R),
        A52_2F2R_LFE => Some(CHMAP_2F2R_LFE),
        A52_3F2R => Some(CHMAP_3F2R),
        A52_3F2R_LFE => Some(CHMAP_3F2R_LFE),
        _ => None,
    }
}

// Fallback to PCM after 9600 bytes without sync, about 50ms.
const FALLBACK_TO_PCM_INTERVAL: i32 = 9600;

struct DecodedState {
    parts_left: usize,
    channel_map: ChannelMap,
    timestamp: time::Time,
}

enum FrameState {
    Skip,
    A52 { flags: c_int },
    A52Decoded(DecodedState),
}

pub struct A52Decoder {
    state: *mut a52_state_t,
    buf: Vec<u8>,
    buf_pos: usize,
    last_sync_pos: i32,
    frame_state: Option<FrameState>,
    frame_size: usize,
    sample_rate: f64,
}

unsafe impl Send for A52Decoder {}

impl A52Decoder {
    pub fn new() -> A52Decoder {
        unsafe {
            A52Decoder {
                state: a52_init(0),
                buf: vec![0u8; 0],
                buf_pos: 0,
                last_sync_pos: 0,
                frame_state: None,
                frame_size: 0,
                sample_rate: 48000f64,
            }
        }
    }

    pub fn add_data(&mut self, data: &[u8], bytes_per_sample: usize) {
        // Cleanup the buffer.
        if self.buf_pos > 16536 {
            self.buf.drain(..self.buf_pos);
            self.last_sync_pos = cmp::max(
                self.last_sync_pos - self.buf_pos as i32,
                -FALLBACK_TO_PCM_INTERVAL,
            );
            self.buf_pos = 0;
        }

        let samples = data.len() / bytes_per_sample;
        self.buf.reserve(samples * 2);
        for i in 0..samples {
            self.buf
                .push(data[i * bytes_per_sample + bytes_per_sample - 1]);
            self.buf
                .push(data[i * bytes_per_sample + bytes_per_sample - 2]);
        }
        self.synchronize();
    }

    fn synchronize(&mut self) {
        while self.frame_state.is_none() && self.buf_pos + HEADER_SIZE <= self.buf.len() {
            if self.buf[self.buf_pos..(self.buf_pos + SPDIF_SYNC_SIZE)] != SPDIF_SYNC {
                self.buf_pos += 1;
                continue;
            }

            let format = self.buf[self.buf_pos + 5] & 0xF;
            if format != SPDIF_AC3 {
                if format != SPDIF_NULL && format != SPDIF_PAUSE {
                    println!("WARNING: unknown SPDIF format: {}", format);
                }

                self.frame_size = 16;
                self.frame_state = Some(FrameState::Skip);
                break;
            }

            // AC3 frame.
            let mut flags: c_int = A52_3F2R | A52_LFE;
            let mut sample_rate: c_int = 0;
            let mut bit_rate: c_int = 0;
            let a52_frame_size = unsafe {
                a52_syncinfo(
                    &(self.buf[self.buf_pos + SPDIF_HEADER_SIZE]),
                    &mut flags,
                    &mut sample_rate,
                    &mut bit_rate,
                )
            };
            if a52_frame_size <= 0 {
                println!("WARNING: Failed to parse A52 header: {}", a52_frame_size);
                self.frame_size = 16;
                self.frame_state = Some(FrameState::Skip);
                break;
            }

            // We have a frame.
            self.sample_rate = sample_rate as f64;
            self.frame_size = SPDIF_HEADER_SIZE + a52_frame_size as usize;
            self.frame_state = Some(FrameState::A52 { flags: flags });
            break;
        }

        if self.frame_state.is_some() {
            self.last_sync_pos = self.buf_pos as i32;
        }
    }

    pub fn have_sync(&self) -> bool {
        self.frame_state.is_some()
    }

    pub fn have_frame(&self) -> bool {
        self.frame_state.is_some() && (self.buf.len() >= self.buf_pos + self.frame_size)
    }

    fn decode_frame(&mut self, flags: c_int) -> Option<ChannelMap> {
        let mut flags: c_int = flags;
        let mut level: sample_t = 1.0;
        let bias: sample_t = 0.0;
        let r = unsafe {
            a52_frame(
                self.state,
                &(self.buf[self.buf_pos + SPDIF_HEADER_SIZE]),
                &mut flags,
                &mut level,
                bias,
            )
        };
        if r < 0 {
            println!("WARNING: Failed to decode A52 frame.");
            return None;
        }

        let channel_map = get_channel_map(flags);
        if channel_map.is_none() {
            println!("WARNING: Unknown channel configuration. flags={}", flags);
        }

        channel_map
    }

    fn get_next_frame(&mut self, state: &DecodedState) -> Option<base::Frame> {
        let mut frame = base::Frame::new(self.sample_rate, state.timestamp, SAMPLES_PER_PART);

        unsafe {
            a52_dynrng(self.state, None, 0 as *mut c_void);

            if a52_block(self.state) != 0 {
                println!("WARNING: A52 decoding failed.");
                return None;
            }

            let mut samples = a52_samples(self.state);
            for &c in state.channel_map {
                frame.ensure_channel(c)[..]
                    .copy_from_slice(slice::from_raw_parts(samples, SAMPLES_PER_PART));
                samples = samples.offset(SAMPLES_PER_PART as isize);
            }
        }

        Some(frame)
    }

    pub fn fallback_to_pcm(&self) -> bool {
        (self.buf_pos as i32 - self.last_sync_pos) > FALLBACK_TO_PCM_INTERVAL
    }

    fn get_frame_internal(&mut self) -> Option<base::Frame> {
        assert!(self.frame_state.is_some());
        if self.buf_pos + self.frame_size > self.buf.len() {
            // Waiting for full frame.
            return None;
        }

        let mut state = match self.frame_state.take() {
            None => return None,
            Some(FrameState::A52 { flags }) => {
                let channel_map = match self.decode_frame(flags) {
                    None => return None,
                    Some(c) => c,
                };

                DecodedState {
                    parts_left: PARTS_PER_FRAME,
                    channel_map: channel_map,
                    timestamp: time::Time::now() - self.delay(),
                }
            }
            Some(FrameState::A52Decoded(state)) => state,
            Some(FrameState::Skip) => return None,
        };

        let frame = match self.get_next_frame(&state) {
            None => return None,
            Some(f) => f,
        };

        state.parts_left -= 1;
        state.timestamp = frame.end_timestamp();
        if state.parts_left > 0 {
            // Keep decoding this frame.
            self.frame_state = Some(FrameState::A52Decoded(state));
        }

        Some(frame)
    }

    pub fn get_frame(&mut self) -> Option<base::Frame> {
        if !self.have_sync() {
            return None;
        }

        let result = self.get_frame_internal();

        if self.frame_state.is_none() {
            // Done with the current frame. Move to the next one.
            self.buf_pos += self.frame_size;
            self.synchronize();
        }

        result
    }

    pub fn delay(&self) -> time::TimeDelta {
        base::samples_to_timedelta(self.sample_rate, SAMPLES_PER_FRAME as i64)
    }
}

impl Drop for A52Decoder {
    fn drop(&mut self) {
        unsafe {
            a52_free(self.state);
        }
    }
}

#[cfg(test)]
mod tests {
    use a52_decoder;
    use std::fs;
    use std::io::Read;

    fn test_file(file: &str, samples_min: usize, samples_max: usize) {
        let mut dec = a52_decoder::A52Decoder::new();
        let mut file = fs::File::open(file).expect(format!("Failed to open {}", file).as_str());

        let mut samples_out = 0;
        loop {
            let mut test_input = vec![0u8; 1024];
            let bytes_read = file
                .read(&mut test_input[..])
                .expect("Failed to read a52-test-input.raw");
            if bytes_read == 0 {
                break;
            }
            dec.add_data(&test_input[..bytes_read], 4);
            'decode_loop: loop {
                assert!(!dec.fallback_to_pcm());
                match dec.get_frame() {
                    None => break 'decode_loop,
                    Some(frame) => {
                        samples_out += frame.len();
                        assert!(frame.channels() == 6);
                    }
                }
            }
        }
        println!("{}", samples_out);
        assert!(samples_out >= samples_min && samples_out <= samples_max);
    }

    #[test]
    fn it_works() {
        test_file("test/a52-test-input.raw", 95000, 97000);
    }

    #[test]
    fn pause() {
        test_file("test/a52-test-input-pause.raw", 0, 0);
    }
    #[test]
    fn pause_start() {
        test_file("test/a52-test-input-pause-start.raw", 95000, 97000);
    }
    #[test]
    fn pause_end() {
        test_file("test/a52-test-input-pause-end.raw", 95000, 97000);
    }
}
