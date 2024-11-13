pub mod detect;

use anyhow::Result;
use image::DynamicImage;
use nalgebra::Point3;

use detect::{BBox, RobotDetector};

#[derive(Debug)]
pub struct RdltResult {
    class_id: u32,
    bbox: BBox,
    confidence: f32,
    location: Point3<f32>,
}

pub struct Radar {
    robot_detector: RobotDetector,
}

impl Radar {
    pub fn new(car_onnx_path: &str, armor_onnx_path: &str) -> Result<Self> {
        Ok(Self {
            robot_detector: RobotDetector::with_defaults(car_onnx_path, armor_onnx_path)?,
        })
    }

    pub fn run_rdlt(
        &self,
        image: &DynamicImage,
        point_cloud: &Vec<Point3<f32>>,
    ) -> Result<Vec<RdltResult>> {
        Ok(self
            .robot_detector
            .detect(&image)?
            .into_iter()
            .map(|det| RdltResult {
                class_id: det.class_id,
                bbox: det.bbox(),
                confidence: det.confidence,
                location: Point3::default(),
            })
            .collect())
    }
}
