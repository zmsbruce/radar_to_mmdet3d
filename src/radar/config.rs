use std::fs;

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{debug, span, trace, Level};

#[derive(Debug, Deserialize)]
pub struct RadarConfig {
    pub detect: DetectorConfig,
    pub locate: LocatorConfig,
    pub instances: Vec<RadarInstanceConfig>,
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

#[derive(Debug, Deserialize)]
pub struct LocatorConfig {
    pub cluster_epsilon: f32,
    pub cluster_min_points: usize,
    pub min_valid_distance: f32,
    pub max_valid_distance: f32,
    pub max_depth_map_queue_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct RadarInstanceConfig {
    pub name: String,
    pub intrinsic: [f32; 9],
    pub lidar_to_camera: [f32; 16],
}

impl RadarConfig {
    pub fn from_file<P>(file_path: P) -> Result<Self>
    where
        P: AsRef<std::path::Path> + std::fmt::Debug,
    {
        let span = span!(Level::TRACE, "RadarConfig::from_file");
        let _enter = span.enter();

        trace!("Reading content from file {:?}...", file_path);
        let config_content =
            fs::read_to_string(file_path).context("Failed to read config from file")?;

        trace!("Deserializing content to RadarConfig...");
        let config: Self = toml::from_str(&config_content)
            .context("Failed to deserialize content to RadarConfig")?;

        debug!("Configurations: {:#?}", config);
        Ok(config)
    }
}
