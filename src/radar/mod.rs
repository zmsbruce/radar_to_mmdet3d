mod config;
mod detect;
mod locate;
mod undistort;

use anyhow::{Context, Result};
use config::RadarConfig;
use image::DynamicImage;
use nalgebra::{Matrix3, Matrix4, Point3, Vector5};
use tracing::{debug, info, span, trace, Level};

use detect::{Execution, RobotDetection, RobotDetector};
use locate::{Locator, RobotLocation};
use undistort::undistort_image;

#[derive(Debug)]
pub struct RobotInfo {
    detection: RobotDetection,
    location: RobotLocation,
}

pub struct RadarInstanceParam {
    camera_intrinsic: Matrix3<f32>,
    distortion: Vector5<f32>,
    lidar_to_camera_transform: Matrix4<f32>,
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

        trace!("Undistorting image...");
        let (image_undistorted, new_camera_matrix) =
            undistort_image(image, &mut instance.camera_intrinsic, &instance.distortion)
                .context("Failed to undistort image")?;

        trace!("Running detection and location...");
        let detect_result = self.robot_detector.detect(&image_undistorted)?;
        let locate_result = self
            .locator
            .locate_detections(
                &point_cloud,
                &detect_result,
                &image_undistorted,
                &instance.lidar_to_camera_transform,
                &new_camera_matrix,
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
