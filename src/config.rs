use std::fs;

use anyhow::Result;
use serde::Deserialize;
use tracing::{debug, error, span, trace, Level};

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
        let config_content = fs::read_to_string(file_path).map_err(|e| {
            error!("Failed to read radar config from file: {e}");
            e
        })?;

        trace!("Deserializing content to RadarConfig...");
        let config: Self = toml::from_str(&config_content).map_err(|e| {
            error!("Failed to parse radar config: {e}");
            e
        })?;

        debug!("Configurations: {:#?}", config);
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
pub struct SourceConfig {
    pub video: Vec<VideoSourceConfig>,
    pub point_cloud_file_path: String,
    pub output_dir_path: String,
}

#[derive(Debug, Deserialize)]
pub struct VideoSourceConfig {
    pub name: String,
    pub file_path: String,
}

impl SourceConfig {
    pub fn from_file<P>(file_path: P) -> Result<Self>
    where
        P: AsRef<std::path::Path> + std::fmt::Debug,
    {
        let span = span!(Level::TRACE, "SourceConfig::from_file");
        let _enter = span.enter();

        trace!("Reading content from file {:?}...", file_path);
        let config_content = fs::read_to_string(file_path).map_err(|e| {
            error!("Failed to read source config from file: {e}");
            e
        })?;

        trace!("Deserializing content to SourceConfig...");
        let config: Self = toml::from_str(&config_content).map_err(|e| {
            error!("Failed to parse source config: {e}");
            e
        })?;

        debug!("Configurations: {:#?}", config);
        Ok(config)
    }
}
