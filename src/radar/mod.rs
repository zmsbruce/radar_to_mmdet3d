mod config;
mod detect;
mod locate;

use std::fmt::Display;

use anyhow::{Context, Result};
use config::{default::*, RadarInstanceConfig};
use image::DynamicImage;
use nalgebra::{Matrix3, Matrix4, Point3};
use tracing::{info, span, trace, Level};

use detect::{Execution, RobotDetection, RobotDetector};
use locate::{Locator, RobotLocation};

#[derive(Debug)]
pub struct RobotInfo {
    detection: RobotDetection,
    location: RobotLocation,
}

impl Display for RobotInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bbox = self.detection.bbox();
        write!(
            f,
            "{}: bbox=(({}, {}), ({}, {})), aabb=(({}, {}, {}), ({}, {}, {})), confidence={}",
            self.detection.label,
            bbox.x_center - bbox.width / 2.0,
            bbox.y_center - bbox.height / 2.0,
            bbox.x_center + bbox.width / 2.0,
            bbox.y_center + bbox.height / 2.0,
            self.location.center.x - self.location.depth / 2.0,
            self.location.center.y - self.location.width / 2.0,
            self.location.center.z - self.location.height / 2.0,
            self.location.center.x + self.location.depth / 2.0,
            self.location.center.y + self.location.width / 2.0,
            self.location.center.z + self.location.height / 2.0,
            self.detection.confidence,
        )
    }
}

pub struct RadarInstance {
    camera_intrinsic: Matrix3<f32>,
    lidar_to_camera_transform: Matrix4<f32>,
}

impl RadarInstance {
    pub fn from_slice(camera_intrinsic: &[f32; 9], lidar_to_camera_transform: &[f32; 16]) -> Self {
        Self {
            camera_intrinsic: Matrix3::from_row_slice(camera_intrinsic),
            lidar_to_camera_transform: Matrix4::from_row_slice(lidar_to_camera_transform),
        }
    }
}

impl From<RadarInstanceConfig> for RadarInstance {
    fn from(value: RadarInstanceConfig) -> Self {
        Self::from_slice(&value.intrinsic, &value.lidar_to_camera)
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
        instance: &RadarInstance,
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

        let robots: Vec<_> = detect_result
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

        robots.iter().for_each(|robot| {
            info!("{robot}");
        });
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
                MIN_VALID_DISTANCE,
                MAX_VALID_DISTANCE,
            ),
        }
    }
}
