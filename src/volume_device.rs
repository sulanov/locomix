use crate::base::*;
use crate::output::Output;
use crate::time::*;
use alsa;
use std::boxed::Box;
use std::ffi::CString;
use toml;

pub trait VolumeDevice: Send {
    // Sets volume to be applied by the device and returns the value that has
    // been applied, which may be adjusted depending on the device.
    fn set_device_gain(&mut self, gain: Gain) -> Gain;
}

pub struct OutputWithVolumeDevice {
    output: Box<dyn Output>,
    volume_device: Box<dyn VolumeDevice>,
    max_level: f32,
    current_gain: Gain,
    device_gain: Gain,
}

impl OutputWithVolumeDevice {
    pub fn new(
        output: Box<dyn Output>,
        volume_device: Box<dyn VolumeDevice>,
    ) -> OutputWithVolumeDevice {
        OutputWithVolumeDevice {
            output,
            volume_device,
            max_level: 1.0,
            current_gain: Gain::zero(),
            device_gain: Gain::zero(),
        }
    }
}

impl Output for OutputWithVolumeDevice {
    fn write(&mut self, mut frame: Frame) -> Result<()> {
        let mut max_level = 1f32;
        for (_c, pcm) in frame.iter_channels() {
            max_level = pcm.iter().fold(max_level, |max, v| {
                let abs = v.abs();
                if abs > max {
                    abs
                } else {
                    max
                }
            });
        }
        let update_max_level = max_level > self.max_level;
        if update_max_level {
            self.max_level = max_level * 1.2;
        }

        if self.current_gain != frame.gain || update_max_level {
            self.current_gain = frame.gain;
            self.device_gain = self
                .volume_device
                .set_device_gain(self.current_gain + Gain::from_level(self.max_level));
        }

        frame.gain -= self.device_gain;

        self.output.write(frame)
    }

    fn deactivate(&mut self) {
        self.max_level = 1.0;
        self.output.deactivate();
    }
    fn sample_rate(&self) -> f64 {
        self.output.sample_rate()
    }
    fn min_delay(&self) -> TimeDelta {
        self.output.min_delay()
    }
}

pub struct AlsaVolume {
    device: String,
    control: String,
}

impl AlsaVolume {
    pub fn new(device: &str, control: &str) -> AlsaVolume {
        AlsaVolume {
            device: device.to_string(),
            control: control.to_string(),
        }
    }

    pub fn create_from_config(config: &toml::value::Table) -> Result<Box<dyn VolumeDevice>> {
        let device = match config.get("device").and_then(|d| d.as_str()) {
            Some(d) => d,
            None => return Err(Error::new("ALSA volume: device string is missing")),
        };

        let control = match config.get("control").and_then(|c| c.as_str()) {
            Some(d) => d,
            None => return Err(Error::new("ALSA volume: control string is missing")),
        };

        Ok(Box::new(AlsaVolume::new(device, control)))
    }

    fn try_set_volume(&mut self, gain: Gain) -> Result<Gain> {
        let mixer = alsa::mixer::Mixer::new(self.device.as_str(), /*nonblock=*/ false)?;
        let mut selem_id = alsa::mixer::SelemId::empty();
        selem_id.set_name(&CString::new(self.control.as_str()).unwrap());
        let selem = match mixer.find_selem(&selem_id) {
            Some(s) => s,
            None => {
                return Err(Error::from_string(format!(
                    "Can fine mixer control {}",
                    self.control
                )))
            }
        };
        selem.set_playback_db_all(alsa::mixer::MilliBel::from_db(gain.db), alsa::Round::Ceil)?;
        let vol = selem.get_playback_vol_db(alsa::mixer::SelemChannelId::FrontLeft)?;
        Ok(Gain { db: vol.to_db() })
    }
}

impl VolumeDevice for AlsaVolume {
    fn set_device_gain(&mut self, gain: Gain) -> Gain {
        match self.try_set_volume(gain) {
            Ok(r) => r,
            Err(e) => {
                println!(
                    "ERROR: Failed to set volume for {} - {}: {:?}",
                    self.device, self.control, e
                );
                Gain::zero()
            }
        }
    }
}
