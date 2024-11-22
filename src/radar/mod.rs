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
                MIN_VALID_DISTANCE,
                MAX_VALID_DISTANCE,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::io::pcd::PcdReader;

    use super::*;

    #[test]
    fn test_radar() -> Result<()> {
        let mut radar = Radar::default();

        let image = image::open(PathBuf::from("assets/test/frame.png"))?;
        let point_cloud: Vec<_> = PcdReader::read_from_file("assets/test/cloud.pcd")?
            .into_iter()
            .map(|v| Point3::new(v[0] as f32, v[1] as f32, v[2] as f32))
            .collect();

        let instance = RadarInstance::from(RadarInstanceConfig {
            name: "Middle".to_string(),
            intrinsic: [
                5314.21858569616,
                0.0,
                1275.27037614003,
                0.0,
                5314.01731877953,
                1029.37643928014,
                0.0,
                0.0,
                1.0,
            ],
            lidar_to_camera: [
                -0.00444956,
                -0.999989,
                0.00122446,
                6.50927,
                0.0173479,
                -0.00130148,
                -0.999849,
                -36.195,
                0.99984,
                -0.00442764,
                0.0173535,
                19.277,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        });

        let result = radar.detect_and_locate(&image, &point_cloud, &instance)?;

        println!("{:#?}", result);

        Ok(())
    }
}
