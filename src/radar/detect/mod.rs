mod yolo;

use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{self, Display},
};

use anyhow::{anyhow, Result};
use image::DynamicImage;
use tracing::{debug, error, span, trace, Level};

pub use yolo::{BBox, Execution};
use yolo::{Detection, Yolo};

use crate::config::DetectorConfig;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum RobotLabel {
    BlueHero,
    BlueEngineer,
    BlueInfantryThree,
    BlueInfantryFour,
    BlueInfantryFive,
    RedHero,
    RedEngineer,
    RedInfantryThree,
    RedInfantryFour,
    RedInfantryFive,
    BlueSentry,
    RedSentry,
}

impl RobotLabel {
    pub fn name(&self) -> &str {
        match self {
            RobotLabel::BlueHero => "Blue Hero",
            RobotLabel::BlueEngineer => "Blue Engineer",
            RobotLabel::BlueInfantryThree => "Blue Infantry Three",
            RobotLabel::BlueInfantryFour => "Blue Infantry Four",
            RobotLabel::BlueInfantryFive => "Blue Infantry Five",
            RobotLabel::RedHero => "Red Hero",
            RobotLabel::RedEngineer => "Red Engineer",
            RobotLabel::RedInfantryThree => "Red Infantry Three",
            RobotLabel::RedInfantryFour => "Red Infantry Four",
            RobotLabel::RedInfantryFive => "Red Infantry Five",
            RobotLabel::BlueSentry => "Blue Sentry",
            RobotLabel::RedSentry => "Red Sentry",
        }
    }

    pub fn name_abbr(&self) -> &str {
        match self {
            RobotLabel::BlueHero => "B1",
            RobotLabel::BlueEngineer => "B2",
            RobotLabel::BlueInfantryThree => "B3",
            RobotLabel::BlueInfantryFour => "B4",
            RobotLabel::BlueInfantryFive => "B5",
            RobotLabel::RedHero => "R1",
            RobotLabel::RedEngineer => "R2",
            RobotLabel::RedInfantryThree => "R3",
            RobotLabel::RedInfantryFour => "R4",
            RobotLabel::RedInfantryFive => "R5",
            RobotLabel::BlueSentry => "B7",
            RobotLabel::RedSentry => "R7",
        }
    }
}

impl Display for RobotLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl TryFrom<u32> for RobotLabel {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(RobotLabel::BlueHero),
            1 => Ok(RobotLabel::BlueEngineer),
            2 => Ok(RobotLabel::BlueInfantryThree),
            3 => Ok(RobotLabel::BlueInfantryFour),
            4 => Ok(RobotLabel::BlueInfantryFive),
            5 => Ok(RobotLabel::RedHero),
            6 => Ok(RobotLabel::RedEngineer),
            7 => Ok(RobotLabel::RedInfantryThree),
            8 => Ok(RobotLabel::RedInfantryFour),
            9 => Ok(RobotLabel::RedInfantryFive),
            10 => Ok(RobotLabel::BlueSentry),
            11 => Ok(RobotLabel::RedSentry),
            _ => Err(anyhow!("Invalid value for RobotLabel")),
        }
    }
}

#[derive(Debug)]
pub struct RobotDetection {
    pub car_detection: Detection,
    pub label: RobotLabel,
    pub confidence: f32,
}

impl RobotDetection {
    pub fn new(car_detection: Detection, armor_detection: Vec<Detection>) -> Option<Self> {
        let span = span!(Level::TRACE, "RobotDetection::new");
        let _enter = span.enter();

        let mut classid_conf_map = HashMap::with_capacity(armor_detection.len());

        trace!("Building class confidence map from armor detections.");
        armor_detection
            .iter()
            .filter(|det| !det.confidence.is_nan())
            .for_each(|det| {
                trace!(
                    "Adding confidence {} for class_id {}.",
                    det.confidence,
                    det.class_id
                );
                *classid_conf_map.entry(det.class_id).or_insert(0.0) += det.confidence
            });

        if let Some(class_id) = classid_conf_map
            .into_iter()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
            .map(|(item, _)| item)
        {
            trace!("Selected class_id {} with highest confidence.", class_id);

            let (sum, count) = armor_detection
                .iter()
                .filter(|det| det.class_id == class_id)
                .map(|det| det.confidence)
                .fold((0.0, 0), |(sum, count), conf| (sum + conf, count + 1));
            let confidence = if count > 0 { sum / (count as f32) } else { 0.0 };

            trace!(
                "Computed average confidence {} for class_id {} over {} detections.",
                confidence,
                class_id,
                count
            );

            Some(Self {
                car_detection,
                label: RobotLabel::try_from(class_id).ok()?,
                confidence,
            })
        } else {
            trace!("No valid class_id found in armor detections.");
            None
        }
    }

    #[inline]
    pub fn bbox(&self) -> BBox {
        self.car_detection.bbox
    }
}

pub struct RobotDetector {
    car_detector: Yolo,
    armor_detector: Yolo,
    execution: Execution,
}

impl RobotDetector {
    pub fn new(
        car_onnx: &str,
        armor_onnx: &str,
        car_conf_thresh: f32,
        armor_conf_thresh: f32,
        car_nms_thresh: f32,
        armor_nms_thresh: f32,
        execution: Execution,
    ) -> Self {
        let span = span!(Level::TRACE, "RobotDetector::new");
        let _enter = span.enter();

        trace!("Initializing car detector...");
        let car_detector = Yolo::new(car_onnx, car_conf_thresh, car_nms_thresh, (640, 640));

        trace!("Initializing armor detector...");
        let armor_detector = Yolo::new(armor_onnx, armor_conf_thresh, armor_nms_thresh, (640, 640));

        debug!("Robot detector initialized.");

        Self {
            car_detector,
            armor_detector,
            execution,
        }
    }

    pub fn from_config(config: &DetectorConfig) -> Result<Self> {
        Ok(RobotDetector::new(
            &config.car_onnx_path,
            &config.armor_onnx_path,
            config.car_conf_thresh,
            config.armor_conf_thresh,
            config.car_nms_thresh,
            config.armor_nms_thresh,
            Execution::try_from(config.execution.as_str()).map_err(|e| {
                error!("Invalid execution {}: {e}", config.execution);
                anyhow!("Invalid execution {}: {e}", config.execution)
            })?,
        ))
    }

    #[inline]
    pub fn is_models_built(&self) -> bool {
        self.car_detector.is_model_built() && self.armor_detector.is_model_built()
    }

    pub fn build_models(&mut self) -> Result<()> {
        self.car_detector.build(self.execution).map_err(|e| {
            error!("Failed to build car detector model: {e}");
            anyhow!("Failed to build car detector model: {e}")
        })?;

        self.armor_detector.build(self.execution).map_err(|e| {
            error!("Failed to build armor detector model: {e}");
            anyhow!("Failed to build armor detector model: {e}")
        })?;

        Ok(())
    }

    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<RobotDetection>> {
        let span = span!(Level::TRACE, "RobotDetector::detect");
        let _enter = span.enter();

        if !self.is_models_built() {
            return Err(anyhow!("Models are not built"));
        }

        trace!("Running car detector inference...");
        let car_detections = self.car_detector.infer(image).map_err(|e| {
            error!("Failed to infer camera image in car detector: {e}");
            anyhow!("Failed to infer camera image in car detector: {e}")
        })?;
        debug!(
            "Car detector inference complete. Detected {} cars.",
            car_detections.len()
        );

        let car_images: Vec<_> = car_detections
            .iter()
            .map(|det| {
                let bbox = &det.bbox;
                debug!(
                    "Cropping car image with bounding box: x_center={}, y_center={}, width={}, height={}.",
                    bbox.x_center,
                    bbox.y_center,
                    bbox.width,
                    bbox.height
                );
                image.crop_imm(
                    (bbox.x_center - bbox.width / 2.0) as u32,
                    (bbox.y_center - bbox.height / 2.0) as u32,
                    bbox.width as u32,
                    bbox.height as u32,
                )
            })
            .collect();

        trace!("Running armor detector inference on cropped car images...");
        let armor_detections = car_images
            .into_iter()
            .map(|car_image| self.armor_detector.infer(&car_image))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                error!("Failed to infer car images in armor detector: {e}");
                anyhow!("Failed to infer car images in armor detector: {e}")
            })?;

        debug!(
            "Armor detector inference complete. Processing {} car-armor pairs.",
            armor_detections.len()
        );

        assert_eq!(car_detections.len(), armor_detections.len());

        let mut robots_map: HashMap<RobotLabel, RobotDetection> =
            HashMap::with_capacity(car_detections.len());
        for (i, (car_det, armor_det)) in car_detections
            .into_iter()
            .zip(armor_detections.into_iter())
            .enumerate()
        {
            debug!(
                "Processing car {} with {} armor detections.",
                i,
                armor_det.len()
            );
            debug!(
                "Car detection: {:?}, armor detection: {:?}",
                car_det, armor_det
            );
            if let Some(robot_det) = RobotDetection::new(car_det, armor_det) {
                debug!(
                    "Car {} classified as label {:?} with confidence {}.",
                    i, robot_det.label, robot_det.confidence
                );
                if let Some(robot_det_exist) = robots_map.get(&robot_det.label) {
                    if robot_det_exist.confidence < robot_det.confidence {
                        debug!(
                            "Updating label {:?} with higher confidence: {} -> {}.",
                            robot_det.label, robot_det_exist.confidence, robot_det.confidence
                        );
                        robots_map.insert(robot_det.label, robot_det);
                    }
                } else {
                    debug!(
                        "Inserting new detection for label {:?} with confidence {}.",
                        robot_det.label, robot_det.confidence
                    );
                    robots_map.insert(robot_det.label, robot_det);
                }
            } else {
                trace!("No valid robot detection for car {}.", i);
            }
        }

        let robots: Vec<_> = robots_map.into_iter().map(|(_k, v)| v).collect();
        debug!("Detection complete. Robots: {:?}.", robots);

        Ok(robots)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::Result;

    use super::*;

    #[test]
    fn test_robot_detector() -> Result<()> {
        let mut robot_detector = RobotDetector::new(
            "assets/test/car.onnx",
            "assets/test/armor.onnx",
            0.30,
            0.45,
            0.50,
            0.75,
            Execution::CPU,
        );

        robot_detector.build_models()?;

        let image = image::open(PathBuf::from("assets/test/battlefield.png"))?;

        let detections = robot_detector.detect(&image)?;
        assert_eq!(detections.len(), 6);
        assert!(detections
            .iter()
            .find(|det| det.label == RobotLabel::RedSentry)
            .is_some());
        assert!(detections
            .iter()
            .find(|det| det.label == RobotLabel::RedEngineer)
            .is_some());
        assert!(detections
            .iter()
            .find(|det| det.label == RobotLabel::RedInfantryFive)
            .is_some());
        assert!(detections
            .iter()
            .find(|det| det.label == RobotLabel::BlueSentry)
            .is_some());
        assert!(detections
            .iter()
            .find(|det| det.label == RobotLabel::BlueEngineer)
            .is_some());
        assert!(detections
            .iter()
            .find(|det| det.label == RobotLabel::BlueInfantryFive)
            .is_some());

        Ok(())
    }
}
