mod config;
mod detect;
mod locate;

use anyhow::{Context, Ok, Result};
use config::RadarConfig;
use image::DynamicImage;
use nalgebra::Point3;

use detect::{BBox, Execution, RobotDetector, RobotLabel};
use tracing::{debug, span, trace, Level};

#[derive(Debug)]
pub struct RdltResult {
    label: RobotLabel,
    bbox: BBox,
    confidence: f32,
    location: Point3<f32>,
}

pub struct Radar {
    robot_detector: RobotDetector,
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

        trace!("Constructing radar...");
        let radar = Self { robot_detector };

        Ok(radar)
    }

    pub fn from_config_file(filename: &str) -> Result<Self> {
        let span = span!(Level::TRACE, "Radar::from_config_file");
        let _enter = span.enter();

        trace!("Reading radar configuration...");
        let radar_config =
            RadarConfig::from_file(filename).context("Failed to initialize Radar")?;

        let detector_config = &radar_config.detect;

        Self::new(
            &detector_config.car_onnx_path,
            &detector_config.armor_onnx_path,
            detector_config.car_conf_thresh,
            detector_config.armor_conf_thresh,
            detector_config.car_nms_thresh,
            detector_config.armor_nms_thresh,
            Execution::try_from(detector_config.execution.as_str())?,
        )
    }

    pub fn run_rdlt(
        &self,
        image: &DynamicImage,
        point_cloud: &Vec<Point3<f32>>,
    ) -> Result<Vec<RdltResult>> {
        let span = span!(Level::TRACE, "Radar::run_rdlt");
        let _enter = span.enter();

        trace!("Running rdlt...");
        let result = self
            .robot_detector
            .detect(&image)?
            .into_iter()
            .map(|det| RdltResult {
                label: det.label,
                bbox: det.bbox(),
                confidence: det.confidence,
                location: Point3::default(),
            })
            .collect();

        debug!("Rdlt result: {:#?}", result);
        Ok(result)
    }
}
