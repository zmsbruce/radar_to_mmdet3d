use std::fs;

use anyhow::{Context, Ok, Result};
use serde::Deserialize;
use tracing::{debug, span, trace, Level};

#[derive(Debug, Deserialize)]
pub struct RadarConfig {
    pub detect: DetectorConfig,
}

#[derive(Debug, Deserialize)]
pub struct DetectorConfig {
    pub car_onnx_path: String,
    pub armor_onnx_path: String,
    pub car_conf_thresh: f32,
    pub armor_conf_thresh: f32,
    pub car_nms_thresh: f32,
    pub armor_nms_thresh: f32,
    pub execution: String,
}

impl RadarConfig {
    pub fn from_file(filename: &str) -> Result<Self> {
        let span = span!(Level::TRACE, "RadarConfig::from_file");
        let _enter = span.enter();

        trace!("Reading content from file {filename}...");
        let config_content =
            fs::read_to_string(filename).context("Failed to read config from file")?;

        trace!("Deserializing content to RadarConfig...");
        let config: Self = toml::from_str(&config_content)
            .context("Failed to deserialize content to RadarConfig")?;

        debug!("Configurations: {:#?}", config);
        Ok(config)
    }
}
