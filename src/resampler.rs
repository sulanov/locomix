use std::cmp;
use std::f32::consts::PI;
use std::sync::Arc;
use base;
use base::convolve;
use time::TimeDelta;

const NUM_SUB_INTERVALS: usize = 32;

#[derive(Clone)]
struct QuadFunction {
    a: f32,
    b: f32,
    c: f32,
}

impl QuadFunction {
    pub fn new(at0: f32, athalf: f32, at1: f32) -> QuadFunction {
        QuadFunction {
            a: 2.0 * at0 + 2.0 * at1 - 4.0 * athalf,
            b: -3.0 * at0 - at1 + 4.0 * athalf,
            c: at0,
        }
    }
}

#[derive(Clone)]
struct QuadSeries {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

struct ResamplerTable {
    kernel: Vec<QuadSeries>,
    size: usize,
}

pub struct Resampler {
    i_freq: f64,
    o_freq: f64,
    queue: Vec<f32>,

    table: Arc<ResamplerTable>,

    i_pos: f64,
    o_pos: f64,
}

fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }
    x.sin() / x
}

fn window(x: f32) -> f32 {
    // Lanczos window.
    if x.abs() > 1.0 {
        0.0
    } else {
        sinc(x * PI)
    }
}

fn kernel(x: f32, size: usize) -> f32 {
    sinc(x * PI) * window(x / (size as f32))
}

impl ResamplerTable {
    pub fn new(size: usize) -> ResamplerTable {
        let mut k = Vec::with_capacity(NUM_SUB_INTERVALS);
        for i in 0..NUM_SUB_INTERVALS {
            let mut s = QuadSeries {
                a: vec![0.0; size * 2],
                b: vec![0.0; size * 2],
                c: vec![0.0; size * 2],
            };
            for p in 0..(size * 2) {
                let x = p as f32 - size as f32 + i as f32 / NUM_SUB_INTERVALS as f32;
                let q = QuadFunction::new(
                        kernel(x, size),
                        kernel(x + 0.5 / NUM_SUB_INTERVALS as f32, size),
                        kernel(x + 1.0 / NUM_SUB_INTERVALS as f32, size),
                    );
                s.a[p] = q.a;
                s.b[p] = q.b;
                s.c[p] = q.c;
            }
            k.push(s);
        }

        ResamplerTable {
            kernel: k,
            size: size,
        }
    }
}

impl Resampler {
    fn new(i_freq: f64, o_freq: f64, table: Arc<ResamplerTable>) -> Resampler {
        let size = table.size;
        Resampler {
            i_freq: i_freq,
            o_freq: o_freq,
            queue: vec![0.0; size],
            table: table,
            i_pos: -(size as f64),
            o_pos: 0.0,
        }
    }

    pub fn clone_state(other: &Resampler) -> Resampler {
        Resampler {
            i_freq: other.i_freq,
            o_freq: other.o_freq,
            queue: vec![0.0; other.queue.len()],
            table: other.table.clone(),
            i_pos: other.i_pos,
            o_pos: other.o_pos,
        }
    }

    pub fn set_frequencies(&mut self, i_freq: f64, o_freq: f64) {
        self.i_pos *= i_freq / self.i_freq;
        self.i_freq = i_freq;

        self.o_pos *= o_freq / self.o_freq;
        self.o_freq = o_freq;
    }

    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output =
            Vec::with_capacity((input.len() as f64 * self.o_freq / self.i_freq) as usize + 1);

        let mut o_pos = self.o_pos;
        let freq_ratio = self.i_freq / self.o_freq;
        loop {
            let o_pos_i_freq = o_pos * freq_ratio - self.i_pos;
            let i_mid = o_pos_i_freq.floor() as usize;

            if i_mid + self.table.size >= self.queue.len() + input.len() {
                break;
            }

            let i_shift = o_pos_i_freq.fract();

            let result = if i_shift == 0.0 {
                if i_mid < self.queue.len() {
                    self.queue[i_mid]
                } else {
                    input[i_mid - self.queue.len()]
                }
            } else {
                let interval = ((1.0 - i_shift) * NUM_SUB_INTERVALS as f64).floor() as usize;
                let interval_pos = ((1.0 - i_shift) * NUM_SUB_INTERVALS as f64).fract();
                let seq = &(self.table.kernel[interval]);

                let start = i_mid - self.table.size + 1;
                let end = i_mid + self.table.size + 1;
                let queue_len = self.queue.len();

                let mut a_sum = 0.0;
                let mut b_sum = 0.0;
                let mut c_sum = 0.0;
                let mut pos;
                let mut i_pos;
                if start < queue_len {
                    pos = queue_len - start;
                    i_pos = 0;
                    a_sum += convolve(&seq.a[0..pos], &self.queue[start..queue_len]);
                    b_sum += convolve(&seq.b[0..pos], &self.queue[start..queue_len]);
                    c_sum += convolve(&seq.c[0..pos], &self.queue[start..queue_len]);
                } else {
                    pos = 0;
                    i_pos = start - queue_len;
                }

                a_sum += convolve(&seq.a[pos..], &input[i_pos..(end - queue_len)]);
                b_sum += convolve(&seq.b[pos..], &input[i_pos..(end - queue_len)]);
                c_sum += convolve(&seq.c[pos..], &input[i_pos..(end - queue_len)]);

                (a_sum as f64 * interval_pos * interval_pos + b_sum as f64 * interval_pos + c_sum as f64) as f32
            };
            output.push(result);
            o_pos += 1.0;
        }

        let samples_to_keep = cmp::min(self.table.size * 2 + 1, self.queue.len() + input.len());
        let samples_to_remove = self.queue.len() + input.len() - samples_to_keep;
        if samples_to_remove > self.queue.len() {
            self.queue.clear();
            self.queue
                .extend_from_slice(&input[(input.len() - samples_to_keep)..]);
        } else {
            self.queue.drain(0..samples_to_remove);
            self.queue.extend_from_slice(input);
        }

        self.i_pos = self.i_pos + samples_to_remove as f64;
        self.o_pos = o_pos;
        while self.i_pos > self.i_freq && self.o_pos > self.o_freq {
            self.i_pos -= self.i_freq;
            self.o_pos -= self.o_freq;
        }

        output
    }
}

pub struct ResamplerFactory {
    table: Arc<ResamplerTable>,
}

impl ResamplerFactory {
    pub fn new(window_size: usize) -> ResamplerFactory {
        ResamplerFactory {
            table: Arc::new(ResamplerTable::new(window_size))
        }
    }

    pub fn create_resampler(&mut self, i_freq: f64, o_freq: f64) -> Resampler {
        Resampler::new(i_freq, o_freq, self.table.clone())
    }
}

pub struct StreamResampler {
    input_sample_rate: usize,
    output_sample_rate: f64,
    reported_output_sample_rate: usize,
    resampler_factory: ResamplerFactory,
    resamplers: Vec<Resampler>,
    delay: TimeDelta,
    window_size: usize,
}

impl StreamResampler {
    pub fn new(
        output_sample_rate: f64,
        reported_output_sample_rate: usize,
        window_size: usize,
    ) -> StreamResampler {
        StreamResampler {
            input_sample_rate: 48000,
            output_sample_rate: output_sample_rate,
            reported_output_sample_rate: reported_output_sample_rate,
            resampler_factory: ResamplerFactory::new(window_size),
            resamplers: Vec::new(),
            delay: TimeDelta::zero(),
            window_size: window_size,
        }
    }

    pub fn get_output_sample_rate(&self) -> f64 {
        self.output_sample_rate
    }

    pub fn set_output_sample_rate(
        &mut self,
        output_sample_rate: f64,
        reported_output_sample_rate: usize,
    ) {
        self.output_sample_rate = output_sample_rate;
        self.reported_output_sample_rate = reported_output_sample_rate;
        for r in self.resamplers.as_mut_slice() {
            r.set_frequencies(self.input_sample_rate as f64, self.output_sample_rate);
        }
    }

    pub fn resample(&mut self, mut frame: base::Frame) -> Option<base::Frame> {
        if self.input_sample_rate != frame.sample_rate {
            self.input_sample_rate = frame.sample_rate;
            for r in self.resamplers.as_mut_slice() {
                r.set_frequencies(self.input_sample_rate as f64, self.output_sample_rate);
            }
            self.delay =
                base::samples_to_timedelta(self.input_sample_rate, self.window_size as i64);
        }

        if self.resamplers.is_empty() {
            self.resamplers.push(self.resampler_factory.create_resampler(
                self.input_sample_rate as f64,
                self.output_sample_rate,
            ));
        }

        while self.resamplers.len() < frame.channels() {
            let new = Resampler::clone_state(&self.resamplers[0]);
            self.resamplers.push(new);
        }

        for i in 0..frame.channels.len() {
            frame.channels[i].pcm = self.resamplers[i].resample(&frame.channels[i].pcm);
        }

        frame.timestamp -= self.delay;
        frame.sample_rate = self.reported_output_sample_rate;

        if frame.len() > 0 {
            Some(frame)
        } else {
            None
        }
    }

    pub fn delay(&self) -> TimeDelta {
        self.delay
    }
}

#[cfg(test)]
mod tests {
    use resampler;
    use std::time::Instant;

    fn val(i: usize, rate: usize, freq: f64) -> f64 {
        let t = i as f64 / rate as f64;
        (t * freq * 2.0 * ::std::f64::consts::PI).sin()
    }

    fn get_power(s: &[f64]) -> f64 {
        s.iter().fold(0f64, |r, &v| r + v * v) / (s.len() as f64)
    }

    #[test]
    fn it_works() {
        let irate: usize = 44100;
        let orate: usize = 96000;
        let f: f64 = 17433.41;

        let mut buf = Vec::<f32>::new();
        for i in 0..(irate * 2) {
            let v = val(i, irate, f);
            // buf.push(((v * 32767.0).round() / 32767.0) as f32);
            buf.push(v as f32);
        }

        let mut factory = resampler::ResamplerFactory::new(200);
        let mut r = factory.create_resampler(irate as f64, orate as f64);
        let mut out = Vec::new();

        let start = Instant::now();

        const RANGE: usize = 441;

        for i in 0..(buf.len() / RANGE) {
            let s = i * RANGE;
            let e = (i + 1) * RANGE;
            out.extend_from_slice(&r.resample(&buf[s..e]));
        }

        let d = Instant::now() - start;
        println!("{} {} ", d.as_secs(), d.subsec_nanos() as f32 / 1000000.0);
        assert!(out.len() > orate as usize - 200);

        let out64: Vec<f64> = out.iter().map(|x| -> f64 { *x as f64 }).collect();

        for i in 1..(out.len() / RANGE) {
            let s = i * RANGE;
            let e = (i + 1) * RANGE;
            let signal = get_power(&out64[s..e]);
            let noise_signal: Vec<f64> =
                ((s + 1)..e).map(|i| out64[i] - val(i, orate, f)).collect();
            let noise = get_power(&noise_signal);
            let nsr = noise / signal;
            let nsr_db = 10.0 * nsr.log(10.0);
            println!("{} {} {} {}", signal, noise, 1.0 / nsr, nsr_db);
            // Target: -96
            assert!(nsr_db < -96.0);
        }
    }
}
