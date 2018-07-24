use base;
use base::convolve;
use std::cmp;
use std::collections::BTreeMap;
use std::f32::consts::PI;
use std::sync::Arc;
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

fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }
    x.sin() / x
}

fn get_window(x: f32) -> f32 {
    // Blackman–Nuttall window
    const A0: f32 = 0.3635819;
    const A1: f32 = 0.4891775;
    const A2: f32 = 0.1365995;
    const A3: f32 = 0.0106411;
    let y = (x as f32 + 1.0) / 2.0;
    A0 - A1 * (2.0 * PI * y).cos() + A2 * (4.0 * PI * y).cos() - A3 * (6.0 * PI * y).cos()
}

fn get_kernel(x: f32, size: f32) -> f32 {
    sinc(x * PI) * get_window(x / size)
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
                let size_f32 = size as f32;
                let x = p as f32 - size_f32 + i as f32 / NUM_SUB_INTERVALS as f32;
                let q = QuadFunction::new(
                    get_kernel(x, size_f32),
                    get_kernel(x + 0.5 / NUM_SUB_INTERVALS as f32, size_f32),
                    get_kernel(x + 1.0 / NUM_SUB_INTERVALS as f32, size_f32),
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

#[derive(Eq, PartialEq, Copy, Clone, Ord, PartialOrd)]
struct FastResamplerConfig {
    freq_num: usize,
    freq_denom: usize,
    size: usize,
}

struct FastResamplerTable {
    config: FastResamplerConfig,
    kernel: Vec<Vec<f32>>,
}

impl FastResamplerTable {
    pub fn new(config: FastResamplerConfig) -> FastResamplerTable {
        let mut kernel = Vec::with_capacity((config.freq_denom + 1) / 2);
        for n in 0..config.freq_denom {
            let mut row = Vec::with_capacity(config.size + 1);
            for i in 0..(config.size * 2) {
                let x = config.size as f32 - i as f32 + n as f32 / config.freq_denom as f32;
                row.push(get_kernel(x, config.size as f32));
            }
            kernel.push(row);
        }

        FastResamplerTable { config, kernel }
    }
}

#[derive(Clone)]
struct FastResampler {
    table: Arc<FastResamplerTable>,

    // Position for the next output sample relative to the
    // first sample in the queue.
    i_pos: usize,
    i_pos_frac: usize,
}

impl FastResampler {
    fn new(table: Arc<FastResamplerTable>) -> FastResampler {
        let i_pos = table.config.size;
        FastResampler {
            table,
            i_pos,
            i_pos_frac: 0,
        }
    }

    fn get_pos(&self) -> f64 {
        self.i_pos as f64 + self.i_pos_frac as f64 / self.table.config.freq_denom as f64
    }

    fn resample(&mut self, input: &[f32], queue: &mut Vec<f32>) -> Vec<f32> {
        let table = &self.table;
        let mut output = Vec::with_capacity(
            (input.len() * table.config.freq_denom / table.config.freq_num) as usize + 1,
        );

        loop {
            if self.i_pos + table.config.size >= queue.len() + input.len() {
                break;
            }

            let mut result: f32 = 0.0;
            let queue_len = queue.len();

            if self.i_pos_frac == 0 {
                result = if self.i_pos < queue_len {
                    queue[self.i_pos]
                } else {
                    input[self.i_pos - queue_len]
                }
            } else {
                let mut start = if self.i_pos + 1 > table.config.size {
                    self.i_pos - table.config.size + 1
                } else {
                    0
                };

                let kernel = &self.table.kernel[self.i_pos_frac];
                let mut k_pos = start + table.config.size - self.i_pos;

                if start < queue_len {
                    result += convolve(
                        &queue[start..queue_len],
                        &kernel[k_pos..(k_pos + queue_len - start)],
                    );
                    k_pos += queue_len - start;
                    start = queue_len;
                }

                let len = self.i_pos + table.config.size - start;
                result += convolve(
                    &input[(start - queue_len)..(self.i_pos + table.config.size - queue_len)],
                    &kernel[k_pos..(k_pos + len)],
                );
            };

            output.push(result);

            self.i_pos_frac += table.config.freq_num;
            self.i_pos += self.i_pos_frac / table.config.freq_denom;
            self.i_pos_frac = self.i_pos_frac % table.config.freq_denom;
        }

        let samples_to_keep = cmp::min(table.config.size * 2 + 1, queue.len() + input.len());
        let samples_to_remove = queue.len() + input.len() - samples_to_keep;
        if samples_to_remove > queue.len() {
            queue.clear();
            queue.extend_from_slice(&input[(input.len() - samples_to_keep)..]);
        } else {
            queue.drain(0..samples_to_remove);
            queue.extend_from_slice(input);
        }

        self.i_pos -= samples_to_remove;

        output
    }
}

pub struct Resampler {
    i_freq: f64,
    o_freq: f64,
    queue: Vec<f32>,

    table: Arc<ResamplerTable>,
    fast_resampler: Option<FastResampler>,

    // Position of the first sample in the queue relative to the input stream,
    i_pos: f64,

    // Position of the next output sample relative to the output stream..
    o_pos: f64,
}

impl Resampler {
    fn new(
        i_freq: f64,
        o_freq: f64,
        table: Arc<ResamplerTable>,
        fast_table: Option<Arc<FastResamplerTable>>,
    ) -> Resampler {
        let size = table.size;
        Resampler {
            i_freq,
            o_freq,
            queue: vec![0.0; size],
            table,
            fast_resampler: fast_table.map(|t| FastResampler::new(t)),
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
            fast_resampler: other.fast_resampler.as_ref().map(|r| r.clone()),
            i_pos: other.i_pos,
            o_pos: other.o_pos,
        }
    }

    pub fn set_frequencies(&mut self, i_freq: f64, o_freq: f64) {
        if self.fast_resampler.is_some() {
            self.i_pos = 0.0;
            self.o_pos = self.fast_resampler.take().unwrap().get_pos() * self.o_freq / self.i_freq;
        }

        self.i_pos *= i_freq / self.i_freq;
        self.i_freq = i_freq;

        self.o_pos *= o_freq / self.o_freq;
        self.o_freq = o_freq;
    }

    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        match self.fast_resampler.as_mut() {
            Some(r) => return r.resample(input, &mut self.queue),
            None => (),
        }

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

                if end > queue_len {
                    a_sum += convolve(&seq.a[pos..], &input[i_pos..(end - queue_len)]);
                    b_sum += convolve(&seq.b[pos..], &input[i_pos..(end - queue_len)]);
                    c_sum += convolve(&seq.c[pos..], &input[i_pos..(end - queue_len)]);
                }

                (a_sum as f64 * interval_pos * interval_pos
                    + b_sum as f64 * interval_pos
                    + c_sum as f64) as f32
            };
            output.push(result);
            o_pos += 1.0;
        }

        let samples_to_keep = cmp::min(self.table.size * 2, self.queue.len() + input.len());
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
    fast_table_cache: BTreeMap<FastResamplerConfig, Arc<FastResamplerTable>>,
    window_size: usize,
}

fn get_greatest_common_divisor(mut a: usize, mut b: usize) -> usize {
    // Euclid's algorithm.
    while b > 0 {
        let c = a % b;
        a = b;
        b = c;
    }
    return a;
}

impl ResamplerFactory {
    pub fn new(window_size: usize) -> ResamplerFactory {
        ResamplerFactory {
            table: Arc::new(ResamplerTable::new(window_size)),
            fast_table_cache: BTreeMap::new(),
            window_size,
        }
    }

    pub fn create_resampler(&mut self, i_freq: f64, o_freq: f64) -> Resampler {
        let fast_table = if i_freq.fract() == 0.0 && o_freq.fract() == 0.0 {
            let i_freq_i = i_freq as usize;
            let o_freq_i = o_freq as usize;
            let gcd = get_greatest_common_divisor(i_freq_i, o_freq_i);
            let freq_num = i_freq_i / gcd;
            let freq_denom = o_freq_i / gcd;

            let config = FastResamplerConfig {
                freq_num: freq_num,
                freq_denom: freq_denom,
                size: self.window_size,
            };

            Some(
                self.fast_table_cache
                    .entry(config)
                    .or_insert_with(|| Arc::new(FastResamplerTable::new(config)))
                    .clone(),
            )
        } else {
            None
        };

        Resampler::new(i_freq, o_freq, self.table.clone(), fast_table)
    }
}

pub struct StreamResampler {
    input_sample_rate: f64,
    output_sample_rate: f64,
    resampler_factory: ResamplerFactory,
    resamplers: base::PerChannel<Resampler>,
    delay: TimeDelta,
    window_size: usize,
}

impl StreamResampler {
    pub fn new(output_sample_rate: f64, window_size: usize) -> StreamResampler {
        StreamResampler {
            input_sample_rate: 48000.0,
            output_sample_rate: output_sample_rate,
            resampler_factory: ResamplerFactory::new(window_size),
            resamplers: base::PerChannel::new(),
            delay: TimeDelta::zero(),
            window_size: window_size,
        }
    }

    fn update_rates(&mut self) {
        for (_, r) in self.resamplers.iter() {
            r.set_frequencies(
                self.input_sample_rate as f64,
                self.output_sample_rate as f64,
            );
        }
    }

    pub fn reset(&mut self) {
        self.resamplers = base::PerChannel::new();
    }

    pub fn set_output_sample_rate(&mut self, output_sample_rate: f64) {
        if self.output_sample_rate == output_sample_rate {
            return;
        }
        self.output_sample_rate = output_sample_rate;
        self.update_rates();
    }

    pub fn resample(&mut self, mut frame: base::Frame) -> Option<base::Frame> {
        if self.input_sample_rate != frame.sample_rate {
            self.input_sample_rate = frame.sample_rate;
            self.update_rates();
            self.delay =
                base::samples_to_timedelta(self.input_sample_rate, self.window_size as i64);
        }

        // Ensure we have resampler for each channel.
        for (c, _) in frame.iter_channels() {
            if !self.resamplers.have_channel(c) {
                let new_resampler = match self.resamplers.iter().next() {
                    None => self
                        .resampler_factory
                        .create_resampler(self.input_sample_rate, self.output_sample_rate),
                    Some((_c, r)) => Resampler::clone_state(r),
                };
                self.resamplers.set(c, new_resampler);
            }
        }

        let mut result = base::Frame::new(self.output_sample_rate, frame.timestamp - self.delay, 0);
        result.gain = frame.gain;

        for (c, pcm) in frame.iter_channels() {
            let pcm = self.resamplers.get_mut(c).unwrap().resample(&pcm);
            if pcm.len() == 0 {
                continue;
            }
            result.set_channel(c, pcm);
        }

        let removed_channels: Vec<base::ChannelPos> = self
            .resamplers
            .iter()
            .map(|(c, _)| c)
            .filter(|c| !frame.have_channel(*c))
            .collect();
        for c in removed_channels {
            self.resamplers.take(c);
        }

        if result.len() > 0 {
            Some(result)
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

        let mut factory = resampler::ResamplerFactory::new(24);
        let mut r = factory.create_resampler(irate as f64, orate as f64);
        let mut out = Vec::new();

        let start = Instant::now();

        const RANGE: usize = 512;

        for i in 0..(buf.len() / RANGE) {
            let s = i * RANGE;
            let e = (i + 1) * RANGE;
            out.extend_from_slice(&r.resample(&buf[s..e]));
            if i == (buf.len() / RANGE / 2) {
                r.set_frequencies(irate as f64, orate as f64);
            }
        }

        let d = Instant::now() - start;
        println!(
            "Time {} {}",
            d.as_secs(),
            d.subsec_nanos() as f32 / 1000000.0,
        );
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
            // Target: -100
            assert!(nsr_db < -100.0);
        }
    }
}
