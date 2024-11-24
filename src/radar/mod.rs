mod config;
mod detect;
mod locate;

use anyhow::{anyhow, Context, Result};
use config::RadarConfig;

use nalgebra::{Matrix3, Matrix4};
use tracing::{error, span, trace, Level};

use detect::{Execution, RobotDetector};
use locate::Locator;

pub struct Radar {
    robot_detector: RobotDetector,
    locator: Locator,
}

impl Radar {
    pub fn new(
        car_onnx: &str,
        armor_onnx: &str,
        car_conf_thresh: f32,
        armor_conf_thresh: f32,
        car_nms_thresh: f32,
        armor_nms_thresh: f32,
        execution: Execution,
        cluster_epsilon: f32,
        cluster_min_points: usize,
        min_valid_distance: f32,
        max_valid_distance: f32,
        lidar_to_camera_transform: Matrix4<f32>,
        camera_intrinsic: Matrix3<f32>,
        max_depth_map_queue_size: usize,
    ) -> Result<Self> {
        let span = span!(Level::TRACE, "Radar::new");
        let _enter = span.enter();

        trace!("Contructing robot detector...");
        let robot_detector = RobotDetector::new(
            car_onnx,
            armor_onnx,
            car_conf_thresh,
            armor_conf_thresh,
            car_nms_thresh,
            armor_nms_thresh,
            execution,
        );
        let locator = Locator::new(
            cluster_epsilon,
            cluster_min_points,
            min_valid_distance,
            max_valid_distance,
            lidar_to_camera_transform,
            camera_intrinsic,
            max_depth_map_queue_size,
        )
        .context("Failed to construct locator")
        .map_err(|e| {
            error!("Failed to construct locator: {e}");
            e
        })?;

        let radar = Self {
            robot_detector,
            locator,
        };
        Ok(radar)
    }

    pub fn from_config_file<P>(file_path: P, instance_name: &str) -> Result<Self>
    where
        P: AsRef<std::path::Path> + std::fmt::Debug,
    {
        let radar_config = RadarConfig::from_file(file_path)
            .context("Failed to open radar config")
            .map_err(|e| {
                error!("Failed to read radar config from file : {}", e);
                e
            })?;

        if let Some(instance) = radar_config
            .instances
            .iter()
            .find(|instance| instance.name == instance_name)
        {
            let execution = Execution::try_from(radar_config.detect.execution.as_str())
                .context("Failed to parse execution")
                .map_err(|e| {
                    error!(
                        "Failed to parse execution {}",
                        radar_config.detect.execution
                    );
                    e
                })?;

            return Radar::new(
                &radar_config.detect.car_onnx_path,
                &radar_config.detect.armor_onnx_path,
                radar_config.detect.car_conf_thresh,
                radar_config.detect.armor_conf_thresh,
                radar_config.detect.car_nms_thresh,
                radar_config.detect.armor_nms_thresh,
                execution,
                radar_config.locate.cluster_epsilon,
                radar_config.locate.cluster_min_points,
                radar_config.locate.min_valid_distance,
                radar_config.locate.max_valid_distance,
                Matrix4::from_row_slice(&instance.lidar_to_camera),
                Matrix3::from_row_slice(&instance.intrinsic),
                radar_config.locate.max_depth_map_queue_size,
            );
        } else {
            error!(
                "{instance_name} is not included in radar instances. Existed: {:?}",
                radar_config.instances
            );
            return Err(anyhow!("Instance name not found"));
        }
    }
}
