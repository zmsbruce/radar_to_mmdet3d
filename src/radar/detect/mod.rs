pub mod config;
mod vis;
mod yolo;

use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{self, Display},
};

use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, GenericImageView};
use raqote::DrawTarget;
use tracing::{debug, info, span, trace, Level};

use vis::{
    display_window_and_waitkey, draw_rect_on_draw_target, draw_text_on_draw_target,
    get_color_from_robot_label,
};
pub use yolo::{BBox, Execution};
use yolo::{Detection, Yolo};

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

impl Display for RobotLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl RobotLabel {
    pub fn as_str(&self) -> &'static str {
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

    pub fn as_str_short(&self) -> &'static str {
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
            RobotLabel::BlueSentry => "Bs",
            RobotLabel::RedSentry => "Rs",
        }
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
    pub armor_detection: Vec<Detection>,
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
                armor_detection,
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
            if let Some(robot_det) = RobotDetection::new(car_det, armor_det) {
                debug!(
                    "Car {} classified as label {:?} with confidence {}.",
                    i + 1,
                    robot_det.label,
                    robot_det.confidence
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
                trace!("No valid robot detection for car {}.", i + 1);
            }
        }

        let robots: Vec<_> = robots_map.into_iter().map(|(_k, v)| v).collect();
        debug!("Detection complete. Robots: {:#?}.", robots);

        Ok(robots)
    }

    #[allow(unused)]
    fn visualize_detections(
        image: &DynamicImage,
        robot_detections: &Vec<RobotDetection>,
    ) -> Result<()> {
        let (width, height) = image.dimensions();
        let mut dt = DrawTarget::new(width as i32, height as i32);

        for detection in robot_detections {
            let bbox = detection.bbox();
            draw_rect_on_draw_target(
                &mut dt,
                &bbox,
                get_color_from_robot_label(detection.label),
                4.,
            );

            let text = format!(
                "{} {:.2}",
                detection.label.as_str_short(),
                detection.confidence
            );
            draw_text_on_draw_target(
                &mut dt,
                &text,
                (
                    bbox.x_center - bbox.width / 2.0,
                    bbox.y_center - bbox.height / 2.0,
                ),
                40.0,
            )
            .context("Failed to draw text")?;

            for armor in &detection.armor_detection {
                let mut bbox = armor.bbox;
                let car_bbox = &detection.bbox();
                bbox.x_center += car_bbox.x_center - car_bbox.width / 2.0;
                bbox.y_center += car_bbox.y_center - car_bbox.height / 2.0;

                draw_rect_on_draw_target(
                    &mut dt,
                    &bbox,
                    get_color_from_robot_label(detection.label),
                    3.,
                );

                let text = format!(
                    "{}: {:.2}",
                    RobotLabel::try_from(armor.class_id)?.as_str_short(),
                    armor.confidence
                );
                draw_text_on_draw_target(
                    &mut dt,
                    &text,
                    (
                        bbox.x_center - bbox.width / 2.0,
                        bbox.y_center - bbox.height / 2.0,
                    ),
                    30.0,
                )
                .context("Failed to draw text")?;
            }
        }

        display_window_and_waitkey(image, &dt).context("Failed to display window")?;

        Ok(())
    }
}
