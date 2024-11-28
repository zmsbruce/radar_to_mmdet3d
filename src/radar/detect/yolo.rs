use std::fmt::Debug;

use anyhow::{anyhow, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{s, Array2, Array4, Axis};
use ort::{
    inputs, CUDAExecutionProvider, GraphOptimizationLevel, OpenVINOExecutionProvider, Session,
    TensorRTExecutionProvider,
};
use tracing::{debug, error, span, trace, warn, Level};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x_center: f32,
    pub y_center: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    pub confidence: f32,
    pub class_id: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum Execution {
    TensorRT,
    CUDA,
    OpenVINO,
    CPU,
    Default,
}

impl TryFrom<&str> for Execution {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "tensorrt" => Ok(Execution::TensorRT),
            "cuda" => Ok(Execution::CUDA),
            "openvino" => Ok(Execution::OpenVINO),
            "cpu" => Ok(Execution::CPU),
            "default" => Ok(Execution::Default),
            _ => Err(anyhow!("Failed to convert {value} to execution")),
        }
    }
}

pub struct Yolo {
    conf_threshold: f32,
    nms_threshold: f32,
    input_size: (u32, u32),
    onnx_path: String,
    model: Option<Session>,
}

impl Debug for Yolo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Yolo")
            .field("conf_threshold", &self.conf_threshold)
            .field("nms_threshold", &self.nms_threshold)
            .field("input_size", &self.input_size)
            .field("onnx_path", &self.onnx_path)
            .field(
                "model",
                &if self.model.is_some() {
                    "Initialized"
                } else {
                    "Unset"
                },
            )
            .finish()
    }
}

impl Yolo {
    pub fn new(
        onnx_path: &str,
        conf_threshold: f32,
        nms_threshold: f32,
        input_size: (u32, u32),
    ) -> Self {
        let span = span!(Level::TRACE, "Yolo::new");
        let _enter = span.enter();

        debug!("Initializing YOLO with config: conf_threshold={}, nms_threshold={}, input_size={:?}, onnx_path={}", 
              conf_threshold, nms_threshold, input_size, onnx_path);

        Self {
            conf_threshold,
            nms_threshold,
            input_size,
            onnx_path: onnx_path.to_string(),
            model: None,
        }
    }

    #[inline]
    pub fn is_model_built(&self) -> bool {
        self.model.is_some()
    }

    pub fn build(&mut self, execution: Execution) -> Result<()> {
        if self.model.is_some() {
            warn!("Yolo {:#?} has already been built.", self);
            return Ok(());
        }

        let span = span!(Level::TRACE, "Yolo::build");
        let _enter = span.enter();

        debug!(
            "Building the ONNX model from onnx: {} and execution: {:?}",
            self.onnx_path, execution
        );
        let providers = match execution {
            Execution::TensorRT => vec![TensorRTExecutionProvider::default().build()],
            Execution::CUDA => vec![CUDAExecutionProvider::default().build()],
            Execution::OpenVINO => vec![OpenVINOExecutionProvider::default().build()],
            Execution::CPU => vec![],
            _ => vec![
                CUDAExecutionProvider::default().build(),
                OpenVINOExecutionProvider::default().build(),
                TensorRTExecutionProvider::default().build(),
            ],
        };

        let session = Session::builder()
            .map_err(|e| {
                error!("Failed to build session builder: {e}");
                e
            })?
            .with_execution_providers(providers)
            .map_err(|e| {
                error!("Failed to registers execution providers: {e}");
                e
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                error!("Failed to set optimization level: {e}");
                e
            })?
            .commit_from_file(&self.onnx_path)
            .map_err(|e| {
                error!("Failed to commit from file: {e}");
                e
            })?;

        self.model = Some(session);

        trace!("ONNX model successfully built.");
        Ok(())
    }

    pub fn infer(&self, image: &DynamicImage) -> Result<Vec<Detection>> {
        let span = span!(Level::TRACE, "Yolo::infer");
        let _enter = span.enter();

        trace!("Starting inference.");

        let (input_tensor, original_dims) = self.preprocess_image(image).map_err(|e| {
            error!("Failed to preprocess image: {e}");
            e
        })?;
        trace!(
            "Image preprocessed with original dimensions = {:?}",
            original_dims
        );

        let model_output = self.run_inference(input_tensor).map_err(|e| {
            error!("Failed to run inference: {e}");
            e
        })?;
        trace!("Inference completed, raw model output received.");

        let detections = self.process_yolov8_output(model_output, original_dims);
        trace!(
            "Processed YOLOv8 output, number of detections: {}",
            detections.len()
        );

        Ok(detections)
    }

    fn preprocess_image(&self, image: &DynamicImage) -> Result<(Array4<f32>, (u32, u32))> {
        let span = span!(Level::TRACE, "Yolo::preprocess_image");
        let _enter = span.enter();

        let original_dims = image.dimensions();
        let (width, height) = self.input_size;

        trace!("Resizing image to {}x{}", width, height);
        let resized_img = image.resize_exact(width, height, FilterType::Nearest);

        let mut input = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
        for (i, pixel) in resized_img
            .as_flat_samples_u8()
            .unwrap()
            .as_slice()
            .chunks(3)
            .enumerate()
        {
            let x = i % width as usize;
            let y = i / width as usize;
            input[[0, 0, y, x]] = pixel[0] as f32 / 255.0; // Red channel
            input[[0, 1, y, x]] = pixel[1] as f32 / 255.0; // Green channel
            input[[0, 2, y, x]] = pixel[2] as f32 / 255.0; // Blue channel
        }

        Ok((input, original_dims))
    }

    fn run_inference(&self, input_tensor: Array4<f32>) -> Result<Array2<f32>> {
        let span = span!(Level::TRACE, "Yolo::run_inference");
        let _enter = span.enter();

        if let Some(model) = &self.model {
            let outputs = model
                .run(inputs!["images" => input_tensor.view()]?)
                .map_err(|e| {
                    error!("Failed to run session: {e}");
                    e
                })?;
            let output = outputs["output0"]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    error!("Failed to extract tensor: {e}");
                    e
                })?
                .t()
                .slice(s![.., .., 0])
                .into_owned();

            trace!("Inference completed successfully.");
            Ok(output)
        } else {
            error!("The ONNX model has not been initialized.");
            Err(anyhow!("The ONNX model has not been initialized."))
        }
    }

    fn process_yolov8_output(&self, output: Array2<f32>, image_size: (u32, u32)) -> Vec<Detection> {
        let span = span!(Level::TRACE, "Yolo::process_yolov8_output");
        let _enter = span.enter();

        let mut detections = Vec::new();

        for row in output.axis_iter(Axis(0)) {
            // Extract bounding box coordinates
            let bbox_x_center = row[0];
            let bbox_y_center = row[1];
            let bbox_width = row[2];
            let bbox_height = row[3];

            // Find the class with the highest confidence score
            let (class_id, confidence) = row
                .iter()
                .skip(4) // Skip bounding box coordinates
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, class_confidence_pair| {
                    if class_confidence_pair.1 > accum.1 {
                        class_confidence_pair
                    } else {
                        accum
                    }
                })
                .unwrap(); // Assuming there's always at least one class

            if confidence < self.conf_threshold {
                continue;
            }

            let (input_width, input_height) = self.input_size;
            let (image_width, image_height) = image_size;

            detections.push(Detection {
                bbox: BBox {
                    x_center: bbox_x_center / (input_width as f32) * (image_width as f32),
                    y_center: bbox_y_center / (input_height as f32) * (image_height as f32),
                    width: bbox_width / (input_width as f32) * (image_width as f32),
                    height: bbox_height / (input_height as f32) * (image_height as f32),
                },
                confidence,
                class_id: class_id as u32,
            });
        }

        trace!("YOLOv8 output processed.");

        let final_detections = self.non_max_suppression(detections);
        trace!("Non-Max Suppression completed.");

        final_detections
    }

    fn non_max_suppression(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
        let span = span!(Level::TRACE, "Yolo::non_max_suppression");
        let _enter = span.enter();

        let mut final_detections = Vec::new();

        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        while !detections.is_empty() {
            let best_detection = detections.remove(0);
            final_detections.push(best_detection.clone());

            detections.retain(|detection| {
                let iou = Self::compute_iou(&best_detection.bbox, &detection.bbox);
                iou < self.nms_threshold
            });
        }

        debug!("Detections: {:#?}.", final_detections);
        final_detections
    }

    fn compute_iou(bbox1: &BBox, bbox2: &BBox) -> f32 {
        let x1_min = bbox1.x_center - bbox1.width / 2.0;
        let y1_min = bbox1.y_center - bbox1.height / 2.0;
        let x1_max = bbox1.x_center + bbox1.width / 2.0;
        let y1_max = bbox1.y_center + bbox1.height / 2.0;

        let x2_min = bbox2.x_center - bbox2.width / 2.0;
        let y2_min = bbox2.y_center - bbox2.height / 2.0;
        let x2_max = bbox2.x_center + bbox2.width / 2.0;
        let y2_max = bbox2.y_center + bbox2.height / 2.0;

        let inter_x_min = x1_min.max(x2_min);
        let inter_y_min = y1_min.max(y2_min);
        let inter_x_max = x1_max.min(x2_max);
        let inter_y_max = y1_max.min(y2_max);

        let inter_area =
            (inter_x_max - inter_x_min).max(0.0) * (inter_y_max - inter_y_min).max(0.0);
        let bbox1_area = (x1_max - x1_min) * (y1_max - y1_min);
        let bbox2_area = (x2_max - x2_min) * (y2_max - y2_min);

        inter_area / (bbox1_area + bbox2_area - inter_area)
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, str::FromStr};

    use super::*;
    use anyhow::Ok;
    use assert_approx_eq::assert_approx_eq;
    use image::RgbImage;

    #[test]
    fn test_preprocess_image() -> Result<()> {
        let img_buffer = RgbImage::from_fn(4, 4, |x, y| {
            if (x + y) % 2 == 0 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 255, 0])
            }
        });

        let yolo = Yolo::new("", 0.0, 0.0, (2, 2));
        let dynamic_image = DynamicImage::ImageRgb8(img_buffer);
        let (processed_array, original_dims) = yolo.preprocess_image(&dynamic_image)?;

        assert_eq!(original_dims, (4, 4));

        let shape = processed_array.shape();
        assert_eq!(shape, &[1, 3, 2, 2]); // batch_size=1, channels=3, height=2, width=2

        let processed_data = processed_array
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to convert {:?} to slice", processed_array))?;

        for &val in processed_data {
            assert!(val >= 0.0 && val <= 1.0, "Pixel value out of range: {val}");
        }

        Ok(())
    }

    #[test]
    fn test_iou_no_overlap() {
        let bbox1 = BBox {
            x_center: 0.0,
            y_center: 0.0,
            width: 2.0,
            height: 2.0,
        };

        let bbox2 = BBox {
            x_center: 5.0,
            y_center: 5.0,
            width: 2.0,
            height: 2.0,
        };

        let iou = Yolo::compute_iou(&bbox1, &bbox2);
        assert_approx_eq!(iou, 0.0);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let bbox1 = BBox {
            x_center: 0.0,
            y_center: 0.0,
            width: 4.0,
            height: 4.0,
        };

        let bbox2 = BBox {
            x_center: 1.0,
            y_center: 1.0,
            width: 4.0,
            height: 4.0,
        };

        let iou = Yolo::compute_iou(&bbox1, &bbox2);
        let expected_iou = 9.0 / (16.0 + 16.0 - 9.0); // intersection area / union area
        assert_approx_eq!(iou, expected_iou);
    }

    #[test]
    fn test_iou_complete_overlap() {
        let bbox1 = BBox {
            x_center: 0.0,
            y_center: 0.0,
            width: 4.0,
            height: 4.0,
        };

        let bbox2 = BBox {
            x_center: 0.0,
            y_center: 0.0,
            width: 4.0,
            height: 4.0,
        };

        let iou = Yolo::compute_iou(&bbox1, &bbox2);
        assert_approx_eq!(iou, 1.0);
    }

    #[test]
    fn test_iou_edge_touching() {
        let bbox1 = BBox {
            x_center: 0.0,
            y_center: 0.0,
            width: 2.0,
            height: 2.0,
        };

        let bbox2 = BBox {
            x_center: 2.0,
            y_center: 0.0,
            width: 2.0,
            height: 2.0,
        };

        let iou = Yolo::compute_iou(&bbox1, &bbox2);
        assert_approx_eq!(iou, 0.0);
    }

    #[test]
    fn test_iou_partial_edge_overlap() {
        let bbox1 = BBox {
            x_center: 0.0,
            y_center: 0.0,
            width: 4.0,
            height: 4.0,
        };

        let bbox2 = BBox {
            x_center: 2.0,
            y_center: 0.0,
            width: 4.0,
            height: 4.0,
        };

        let iou = Yolo::compute_iou(&bbox1, &bbox2);
        let expected_iou = 8.0 / (16.0 + 16.0 - 8.0); // intersection area / union area
        assert_approx_eq!(iou, expected_iou);
    }

    #[test]
    fn test_non_max_suppression() {
        let detection1 = Detection {
            bbox: BBox {
                x_center: 0.5,
                y_center: 0.5,
                width: 0.2,
                height: 0.2,
            },
            confidence: 0.9,
            class_id: 0,
        };
        let detection2 = Detection {
            bbox: BBox {
                x_center: 0.52,
                y_center: 0.52,
                width: 0.2,
                height: 0.2,
            },
            confidence: 0.8,
            class_id: 0,
        };
        let detection3 = Detection {
            bbox: BBox {
                x_center: 0.7,
                y_center: 0.7,
                width: 0.2,
                height: 0.2,
            },
            confidence: 0.7,
            class_id: 0,
        };

        let detections = vec![detection1, detection2, detection3];

        let nms_threshold = 0.3;
        let yolo = Yolo::new("", 0.0, nms_threshold, (0, 0));
        let final_detections = yolo.non_max_suppression(detections);

        assert_eq!(
            final_detections.len(),
            2,
            "Incorrect length of final detections"
        );

        assert!(
            final_detections[0].confidence > final_detections[1].confidence,
            "Incorrect confidence sort"
        );
    }

    fn create_mock_yolov8_output() -> Array2<f32> {
        let num_predictions = 5;
        let num_classes = 80;
        let mut output = Array2::zeros((num_predictions, num_predictions + num_classes)); // [5, 85] 张量

        output[[0, 0]] = 0.7; // x_center
        output[[0, 1]] = 0.5; // y_center
        output[[0, 2]] = 0.2; // width
        output[[0, 3]] = 0.2; // height

        output[[1, 0]] = 0.3; // x_center
        output[[1, 1]] = 0.6; // y_center
        output[[1, 2]] = 0.2; // width
        output[[1, 3]] = 0.2; // height

        output[[0, 4]] = 0.9; // confidence
        output[[1, 5]] = 0.8;

        output
    }

    #[test]
    fn test_process_yolov8_output() {
        let mock_output = create_mock_yolov8_output();

        let conf_threshold = 0.5;
        let nms_threshold = 0.4;
        let yolo = Yolo::new("", conf_threshold, nms_threshold, (1, 1));

        let detections = yolo.process_yolov8_output(mock_output, (1, 1));

        assert_eq!(detections.len(), 2, "Incorrect size of detections");

        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[1].class_id, 1);

        assert!(detections[0].bbox.x_center > 0.5);
        assert!(detections[1].bbox.x_center < 0.5);
    }

    #[test]
    fn test_yolo() -> Result<()> {
        let mut yolo = Yolo::new("assets/test/yolov8n.onnx", 0.5, 0.75, (640, 640));
        yolo.build(Execution::CPU)?;

        let img = image::open(PathBuf::from_str("assets/test/zidane.jpg")?)?;
        let detections = yolo.infer(&img)?;

        assert!(detections.len() > 0);

        Ok(())
    }
}
