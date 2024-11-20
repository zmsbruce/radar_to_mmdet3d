mod config;
mod detect;
mod locate;
mod undistort;

use anyhow::{Context, Result};
use config::RadarConfig;
use image::DynamicImage;
use nalgebra::{Matrix3, Matrix4, Point3, Vector5};
use tracing::{debug, span, trace, Level};

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
    lidar_to_world_transform: Matrix4<f32>,
    lidar_to_camera_transform: Matrix4<f32>,
}

pub struct Radar {
    robot_detector: RobotDetector,
    locator: Locator,
}

impl Radar {
    pub fn new(
        car_onnx_path: &str,
        armor_onnx_path: &str,
        car_conf_thresh: f32,
        armor_conf_thresh: f32,
        car_nms_thresh: f32,
        armor_nms_thresh: f32,
        execution: Execution,
        cluster_epsilon: f32,
        cluster_min_points: usize,
        min_distance_to_background: f32,
        max_distance_to_background: f32,
        max_valid_distance: f32,
        depth_map_queue_size: usize,
    ) -> Result<Self> {
        let span = span!(Level::TRACE, "Radar::new");
        let _enter = span.enter();

        trace!("Constructing robot detector...");
        debug!("Robot detector params: car_onnx_path = {}, armor_onnx_path = {}, car_conf_thresh = {}, armor_conf_thresh = {}, car_nms_thresh = {}, armor_nms_thresh = {}, execution = {:?}", 
               car_onnx_path, armor_onnx_path, car_conf_thresh, armor_conf_thresh, car_nms_thresh, armor_nms_thresh, execution);

        let robot_detector = RobotDetector::new(
            car_onnx_path,
            armor_onnx_path,
            car_conf_thresh,
            armor_conf_thresh,
            car_nms_thresh,
            armor_nms_thresh,
            execution,
        )
        .context("Failed to construct robot detector")?;

        let locator = Locator::new(
            cluster_epsilon,
            cluster_min_points,
            min_distance_to_background,
            max_distance_to_background,
            max_valid_distance,
            depth_map_queue_size,
        );

        trace!("Constructing radar...");
        let radar = Self {
            robot_detector,
            locator,
        };

        Ok(radar)
    }

    pub fn from_config_file(filename: &str) -> Result<Self> {
        let span = span!(Level::TRACE, "Radar::from_config_file");
        let _enter = span.enter();

        trace!("Reading radar configuration...");
        let radar_config =
            RadarConfig::from_file(filename).context("Failed to initialize Radar")?;

        Self::new(
            &radar_config.detect.car_onnx_path,
            &radar_config.detect.armor_onnx_path,
            radar_config.detect.car_conf_thresh,
            radar_config.detect.armor_conf_thresh,
            radar_config.detect.car_nms_thresh,
            radar_config.detect.armor_nms_thresh,
            Execution::try_from(radar_config.detect.execution.as_str())?,
            radar_config.locate.cluster_epsilon,
            radar_config.locate.cluster_min_points,
            radar_config.locate.min_distance_to_background,
            radar_config.locate.max_distance_to_background,
            radar_config.locate.max_valid_distance,
            radar_config.locate.depth_map_queue_size,
        )
    }

    pub fn detect_and_locate(
        &mut self,
        image: &DynamicImage,
        point_cloud: &Vec<Point3<f32>>,
        instance: &mut RadarInstanceParam,
    ) -> Result<Vec<RobotInfo>> {
        let span = span!(Level::TRACE, "Radar::detect_and_locate");
        let _enter = span.enter();

        trace!("Undistorting image...");
        let image_undistorted =
            undistort_image(image, &mut instance.camera_intrinsic, &instance.distortion)
                .context("Failed to undistort image")?;

        trace!("Running detection and location...");
        let detect_result = self.robot_detector.detect(&image_undistorted)?;
        let locate_result = self.locator.locate_detections(
            &point_cloud,
            &detect_result,
            &image_undistorted,
            &instance.lidar_to_world_transform,
            &instance.lidar_to_camera_transform,
            &instance.camera_intrinsic,
        )?;

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
