mod config;
mod detect;
mod locate;

use anyhow::{Context, Result};
use config::{default::*, RadarInstanceConfig};
use image::DynamicImage;
use nalgebra::{Matrix3, Matrix4, Point3};
use tracing::{debug, info, span, trace, Level};

use detect::{Execution, RobotDetection, RobotDetector};
use locate::{Locator, RobotLocation};

#[derive(Debug)]
pub struct RobotInfo {
    detection: RobotDetection,
    location: RobotLocation,
}

pub struct RadarInstanceParam {
    camera_intrinsic: Matrix3<f32>,
    lidar_to_camera_transform: Matrix4<f32>,
}

impl From<RadarInstanceConfig> for RadarInstanceParam {
    fn from(value: RadarInstanceConfig) -> Self {
        RadarInstanceParam {
            camera_intrinsic: Matrix3::from_row_slice(&value.intrinsic),
            lidar_to_camera_transform: Matrix4::from_row_slice(&value.lidar_to_camera),
        }
    }
}

pub struct Radar<'a> {
    robot_detector: RobotDetector<'a>,
    locator: Locator,
}

impl<'a> Radar<'a> {
    pub fn detect_and_locate(
        &mut self,
        image: &DynamicImage,
        point_cloud: &Vec<Point3<f32>>,
        instance: &mut RadarInstanceParam,
    ) -> Result<Vec<RobotInfo>> {
        let span = span!(Level::TRACE, "Radar::detect_and_locate");
        let _enter = span.enter();

        if !self.robot_detector.is_models_built() {
            info!("Building car and armor models...");
            self.robot_detector
                .build_models(Execution::Default)
                .context("Failed to build models")?;
        }

        trace!("Running detection and location...");
        let detect_result = self.robot_detector.detect(&image)?;
        let locate_result = self
            .locator
            .locate_detections(
                &point_cloud,
                &detect_result,
                &image,
                &instance.lidar_to_camera_transform,
                &instance.camera_intrinsic,
            )
            .context("Failed to locate detections")?;

        let robots = detect_result
            .into_iter()
            .zip(locate_result.into_iter())
            .filter_map(|(detection, location)| {
                if let Some(location) = location {
                    Some(RobotInfo {
                        detection,
                        location,
                    })
                } else {
                    None
                }
            })
            .collect();

        debug!("Rdlt result: {:#?}", robots);
        Ok(robots)
    }
}

impl Default for Radar<'_> {
    fn default() -> Self {
        Self {
            robot_detector: RobotDetector::new(
                include_bytes!("../../assets/model/car.onnx"),
                include_bytes!("../../assets/model/armor.onnx"),
                CAR_CONF_THRESH,
                ARMOR_CONF_THRESH,
                CAR_NMS_THRESH,
                ARMOR_NMS_THRESH,
            ),
            locator: Locator::new(
                CLUSTER_EPSILON,
                CLUSTER_MIN_POINTS,
                MIN_DISTANCE_TO_BACKGROUND,
                MAX_DISTANCE_TO_BACKGROUND,
                MAX_VALID_DISTANCE,
            ),
        }
    }
}
