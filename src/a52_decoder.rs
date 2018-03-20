#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

extern crate libc;
use self::libc::{c_int, c_void, uint32_t, uint8_t};
use base;
use time;
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

static CHMAP_MONO: &'static [base::ChannelPos] = &[base::ChannelPos::FC];
static CHMAP_MONO_LFE: &'static [base::ChannelPos] = &[base::ChannelPos::Sub, base::ChannelPos::FC];

static CHMAP_STEREO: &'static [base::ChannelPos] = &[base::ChannelPos::FL, base::ChannelPos::FR];
static CHMAP_STEREO_LFE: &'static [base::ChannelPos] = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FR,
];

static CHMAP_3F: &'static [base::ChannelPos] = &[
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
];
static CHMAP_3F_LFE: &'static [base::ChannelPos] = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
];

static CHMAP_2F1R: &'static [base::ChannelPos] = &[
    base::ChannelPos::FL,
    base::ChannelPos::FR,
    base::ChannelPos::SC,
];
static CHMAP_2F1R_LFE: &'static [base::ChannelPos] = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FR,
    base::ChannelPos::SC,
];
static CHMAP_3F1R: &'static [base::ChannelPos] = &[
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
    base::ChannelPos::SC,
];
static CHMAP_3F1R_LFE: &'static [base::ChannelPos] = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
    base::ChannelPos::SC,
];
static CHMAP_2F2R: &'static [base::ChannelPos] = &[
    base::ChannelPos::FL,
    base::ChannelPos::FR,
    base::ChannelPos::SL,
    base::ChannelPos::SR,
];
static CHMAP_2F2R_LFE: &'static [base::ChannelPos] = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FR,
    base::ChannelPos::SL,
    base::ChannelPos::SR,
];
static CHMAP_3F2R: &'static [base::ChannelPos] = &[
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
    base::ChannelPos::SL,
    base::ChannelPos::SR,
];
static CHMAP_3F2R_LFE: &'static [base::ChannelPos] = &[
    base::ChannelPos::Sub,
    base::ChannelPos::FL,
    base::ChannelPos::FC,
    base::ChannelPos::FR,
    base::ChannelPos::SL,
    base::ChannelPos::SR,
];

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

pub enum DecodeResult {
    FallbackToPcm,
    Decoded(base::Frame),
    NoFrame,
}

enum FrameType {
    Skip,
    A52 { flags: c_int, sample_rate: usize },
}

struct FrameInfo {
    frame_size: usize,
    type_: FrameType,
}

pub struct A52Decoder {
    state: *mut a52_state_t,
    buf: Vec<u8>,
    buf_pos: usize,
    last_sync_pos: i32,
    frame_info: Option<FrameInfo>,
}

unsafe impl Send for A52Decoder {}

impl A52Decoder {
    pub fn new() -> A52Decoder {
        unsafe {
            A52Decoder {
                state: a52_init(0),
                buf: vec![0],
                buf_pos: 0,
                last_sync_pos: 0,
                frame_info: None,
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
        while self.frame_info.is_none() && self.buf_pos + HEADER_SIZE <= self.buf.len() {
            if self.buf[self.buf_pos..(self.buf_pos + SPDIF_SYNC_SIZE)] != SPDIF_SYNC {
                self.buf_pos += 1;
                continue;
            }

            let format = self.buf[self.buf_pos + 5] & 0xF;
            if format != SPDIF_AC3 {
                if format != SPDIF_NULL && format != SPDIF_PAUSE {
                    println!("WARNING: unknown SPDIF format: {}", format);
                }

                self.frame_info = Some(FrameInfo {
                    frame_size: 16,
                    type_: FrameType::Skip,
                });
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
                self.frame_info = Some(FrameInfo {
                    frame_size: 16,
                    type_: FrameType::Skip,
                });
                break;
            }

            // We have a frame.
            self.frame_info = Some(FrameInfo {
                frame_size: SPDIF_HEADER_SIZE + a52_frame_size as usize,
                type_: FrameType::A52 {
                    flags: flags,
                    sample_rate: sample_rate as usize,
                },
            });
            break;
        }

        if self.frame_info.is_some() {
            self.last_sync_pos = self.buf_pos as i32;
        }
    }

    pub fn have_sync(&self) -> bool {
        self.frame_info.is_some()
    }

    pub fn have_frame(&self) -> bool {
        match self.frame_info.as_ref() {
            None => false,
            Some(info) => self.buf.len() >= self.buf_pos + info.frame_size,
        }
    }

    fn decode_frame(&mut self, flags: c_int, sample_rate: usize) -> DecodeResult {
        let mut frame = base::Frame::new(sample_rate as f32, time::Time::now(), 6 * 256);

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
            return DecodeResult::NoFrame;
        }

        let channel_map = match get_channel_map(flags) {
            None => {
                println!("WARNING: Unknown channel configuration. flags={}", flags);
                return DecodeResult::NoFrame;
            }
            Some(map) => map,
        };

        for i in 0..6 {
            unsafe {
                a52_dynrng(self.state, None, 0 as *mut c_void);

                if a52_block(self.state) != 0 {
                    println!("WARNING: A52 decoding failed.");
                    return DecodeResult::NoFrame;
                }

                let mut samples = a52_samples(self.state);
                for c in channel_map {
                    frame.ensure_channel(*c)[i * 256..(i + 1) * 256]
                        .copy_from_slice(slice::from_raw_parts(samples, 256));
                    samples = samples.offset(256);
                }
            }
        }

        DecodeResult::Decoded(frame)
    }

    pub fn get_frame(&mut self) -> DecodeResult {
        if !self.have_frame() {
            if (self.buf_pos as i32 - self.last_sync_pos) > FALLBACK_TO_PCM_INTERVAL {
                return DecodeResult::FallbackToPcm;
            } else {
                return DecodeResult::NoFrame;
            }
        }

        let frame_info = self.frame_info.take().unwrap();

        let result = match frame_info.type_ {
            FrameType::A52 { flags, sample_rate } => self.decode_frame(flags, sample_rate),
            FrameType::Skip => DecodeResult::NoFrame,
        };

        self.buf_pos += frame_info.frame_size;
        self.synchronize();

        result
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
    use base;
    use std::fs;
    use std::io::{Read, Write};

    fn test_file(file: &str, samples_min: usize, samples_max: usize) {
        let mut dec = a52_decoder::A52Decoder::new();
        let mut file = fs::File::open(file).expect(format!("Failed to open {}", file).as_str());

        let mut samples_out = 0;
        loop {
            let mut test_input = vec![0u8; 1024];
            let bytes_read = file.read(&mut test_input[..])
                .expect("Failed to read a52-test-input.raw");
            if bytes_read == 0 {
                break;
            }
            dec.add_data(&test_input[..bytes_read], 4);
            'decode_loop: loop {
                match dec.get_frame() {
                    a52_decoder::DecodeResult::FallbackToPcm => assert!(false),
                    a52_decoder::DecodeResult::NoFrame => break 'decode_loop,
                    a52_decoder::DecodeResult::Decoded(frame) => {
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
