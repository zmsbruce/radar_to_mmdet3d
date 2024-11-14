mod yolo;

use std::{cmp::Ordering, collections::HashMap};

use anyhow::Result;
use image::DynamicImage;
use tracing::{debug, info, span, trace, Level};

pub use yolo::{BBox, Execution};
use yolo::{Detection, Yolo};

#[derive(Debug)]
pub struct RobotDetection {
    pub armor_detection: Vec<Detection>,
    pub car_detection: Detection,
    pub class_id: u32,
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
                armor_detection,
                car_detection,
                class_id,
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
}

impl RobotDetector {
    pub fn new(
        car_onnx_path: &str,
        armor_onnx_path: &str,
        car_conf_thresh: f32,
        armor_conf_thresh: f32,
        car_nms_thresh: f32,
        armor_nms_thresh: f32,
        execution: Execution,
    ) -> Result<Self> {
        let span = span!(Level::TRACE, "RobotDetector::new");
        let _enter = span.enter();

        info!("Initializing car detector...");
        let car_detector = Yolo::new(car_onnx_path, car_conf_thresh, car_nms_thresh, (640, 640))
            .build(execution)?;

        info!("Initializing armor detector...");
        let armor_detector = Yolo::new(
            armor_onnx_path,
            armor_conf_thresh,
            armor_nms_thresh,
            (640, 640),
        )
        .build(execution)?;

        info!("Robot detector initialized.");
        Ok(Self {
            car_detector,
            armor_detector,
        })
    }

    pub fn with_defaults(car_onnx_path: &str, armor_onnx_path: &str) -> Result<Self> {
        const DEFAULT_CAR_CONF_THRESH: f32 = 0.40;
        const DEFAULT_ARMOR_CONF_THRESH: f32 = 0.65;
        const DEFAULT_CAR_NMS_THRESH: f32 = 0.50;
        const DEFAULT_ARMOR_NMS_THRESH: f32 = 0.75;

        RobotDetector::new(
            car_onnx_path,
            armor_onnx_path,
            DEFAULT_CAR_CONF_THRESH,
            DEFAULT_ARMOR_CONF_THRESH,
            DEFAULT_CAR_NMS_THRESH,
            DEFAULT_ARMOR_NMS_THRESH,
            Execution::Default,
        )
    }

    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<RobotDetection>> {
        let span = span!(Level::TRACE, "RobotDetector::detect");
        let _enter = span.enter();

        assert!(self.car_detector.is_session_built() && self.armor_detector.is_session_built());

        trace!("Running car detector inference...");
        let car_detections = self.car_detector.infer(image)?;
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
        let armor_detections: Vec<_> = car_images
            .into_iter()
            .map(|image| -> Result<_> {
                let armor_detection = self.armor_detector.infer(&image)?;
                debug!(
                    "Armor detector detected {} objects in cropped car image.",
                    armor_detection.len()
                );
                Ok(armor_detection)
            })
            .collect::<Result<Vec<_>, _>>()?;
        debug!(
            "Armor detector inference complete. Processing {} car-armor pairs.",
            armor_detections.len()
        );

        assert_eq!(car_detections.len(), armor_detections.len());

        let mut robots_map: HashMap<u32, RobotDetection> =
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
            if let Some(robot_det) = RobotDetection::new(car_det, armor_det) {
                debug!(
                    "Car {} classified as class_id {} with confidence {}.",
                    i + 1,
                    robot_det.class_id,
                    robot_det.confidence
                );
                if let Some(robot_det_exist) = robots_map.get(&robot_det.class_id) {
                    if robot_det_exist.confidence < robot_det.confidence {
                        debug!(
                            "Updating class_id {} with higher confidence: {} -> {}.",
                            robot_det.class_id, robot_det_exist.confidence, robot_det.confidence
                        );
                        robots_map.insert(robot_det.class_id, robot_det);
                    }
                } else {
                    debug!(
                        "Inserting new detection for class_id {} with confidence {}.",
                        robot_det.class_id, robot_det.confidence
                    );
                    robots_map.insert(robot_det.class_id, robot_det);
                }
            } else {
                trace!("No valid robot detection for car {}.", i + 1);
            }
        }

        let robots: Vec<_> = robots_map.into_iter().map(|(_k, v)| v).collect();
        debug!("Detection complete. Robots: {:#?}.", robots);

        Ok(robots)
    }
}
