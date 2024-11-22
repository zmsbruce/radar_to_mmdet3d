use std::fs;

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{debug, span, trace, Level};

pub mod default {
    pub const CAR_CONF_THRESH: f32 = 0.30;
    pub const ARMOR_CONF_THRESH: f32 = 0.65;

    pub const CAR_NMS_THRESH: f32 = 0.50;
    pub const ARMOR_NMS_THRESH: f32 = 0.75;

    pub const CLUSTER_EPSILON: f32 = 0.3;
    pub const CLUSTER_MIN_POINTS: usize = 8;

    pub const MIN_DISTANCE_TO_BACKGROUND: f32 = 0.2;
    pub const MAX_DISTANCE_TO_BACKGROUND: f32 = 8.0;

    pub const MAX_VALID_DISTANCE: f32 = 29.3;
}

#[derive(Debug, Deserialize)]
pub struct RadarInstanceConfig {
    pub name: String,
    pub intrinsic: [f32; 9],
    pub lidar_to_camera: [f32; 16],
}

#[derive(Debug, Deserialize)]
pub struct RadarInstanceArrayConfig {
    pub instances: Vec<RadarInstanceConfig>,
}

impl RadarInstanceArrayConfig {
    pub fn from_file(filename: &str) -> Result<Self> {
        let span = span!(Level::TRACE, "RadarInstancesConfig::from_file");
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
