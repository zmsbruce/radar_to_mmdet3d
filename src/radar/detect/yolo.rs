use anyhow::{anyhow, Context, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{s, Array2, Array4, Axis};
use ort::{
    inputs, CUDAExecutionProvider, GraphOptimizationLevel, OpenVINOExecutionProvider, Session,
    TensorRTExecutionProvider,
};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use rusttype::{point, Font, Scale};
use show_image::{
    event::{VirtualKeyCode, WindowEvent},
    AsImageView, WindowOptions,
};
use tracing::{debug, error, info, span, trace, Level};

#[derive(Debug, Clone, PartialEq)]
pub struct BBox {
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    bbox: BBox,
    confidence: f32,
    class_id: u32,
}

pub struct Yolo {
    conf_threshold: f32,
    nms_threshold: f32,
    input_size: (u32, u32),
    onnx_path: String,
    model: Option<Session>,
}

#[derive(Debug)]
pub enum Execution {
    TensorRT,
    CUDA,
    OpenVINO,
    CPU,
    Default,
}

impl Yolo {
    pub fn new(
        conf_threshold: f32,
        nms_threshold: f32,
        input_size: (u32, u32),
        onnx_path: &str,
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

    pub fn build(mut self, execution: Execution) -> Result<Self> {
        let span = span!(Level::TRACE, "Yolo::build");
        let _enter = span.enter();

        info!(
            "Building the ONNX model from file: {} and execution: {:?}",
            self.onnx_path, execution
        );
        let providers = match execution {
            Execution::TensorRT => vec![TensorRTExecutionProvider::default().build()],
            Execution::CUDA => vec![CUDAExecutionProvider::default().build()],
            Execution::OpenVINO => vec![OpenVINOExecutionProvider::default().build()],
            Execution::CPU => vec![],
            _ => vec![
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                OpenVINOExecutionProvider::default().build(),
            ],
        };

        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_execution_providers(providers)
            .context("Failed to register execution providers")?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&self.onnx_path)
            .context("Failed to commit session from file")?;

        self.model = Some(session);

        trace!("ONNX model successfully built.");
        Ok(self)
    }

    pub fn infer(&self, image: &DynamicImage) -> Result<Vec<Detection>> {
        let span = span!(Level::TRACE, "Yolo::infer");
        let _enter = span.enter();

        trace!("Starting inference.");

        let (input_tensor, original_dims) = self
            .preprocess_image(image)
            .context("Failed to preprocess image")?;
        trace!(
            "Image preprocessed with original dimensions = {:?}",
            original_dims
        );

        let model_output = self
            .run_inference(input_tensor)
            .context("Failed to run inference")?;
        trace!("Inference completed, raw model output received.");

        let detections = self.process_yolov8_output(model_output, original_dims);
        trace!(
            "Processed YOLOv8 output, number of detections: {}",
            detections.len()
        );

        Ok(detections)
    }

    #[allow(unused)]
    pub fn visualize(img: &DynamicImage, dets: &Vec<Detection>) -> Result<()> {
        let (width, height) = img.dimensions();
        let mut dt = DrawTarget::new(width as i32, height as i32);

        let font_data = include_bytes!("../../../assets/NotoSans-Regular.ttf") as &[u8];
        let font = Font::try_from_bytes(font_data).context("Failed to load font")?;
        let font_scale = Scale::uniform(40.0);
        let v_metrics = font.v_metrics(font_scale);

        for det in dets {
            let bbox = &det.bbox;

            let mut pb = PathBuilder::new();
            pb.rect(
                bbox.x_center - bbox.width / 2.0,
                bbox.y_center - bbox.height / 2.0,
                bbox.width,
                bbox.height,
            );
            let path = pb.finish();

            let color = Self::get_color_for_class(det.class_id as usize);
            dt.stroke(
                &path,
                &Source::Solid(color),
                &StrokeStyle {
                    join: LineJoin::Round,
                    width: 4.,
                    ..StrokeStyle::default()
                },
                &DrawOptions::default(),
            );

            let text = format!("{}: {:.2}", det.class_id, det.confidence);

            let text_x = bbox.x_center - bbox.width / 2.0;
            let text_y = bbox.y_center - bbox.height / 2.0;
            let offset = point(text_x + 10.0, text_y + v_metrics.ascent);

            let glyphs: Vec<_> = font.layout(&text, font_scale, offset).collect();
            for glyph in glyphs {
                if let Some(bbox) = glyph.pixel_bounding_box() {
                    glyph.draw(|x, y, v| {
                        dt.fill_rect(
                            (x as i32 + bbox.min.x) as f32,
                            (y as i32 + bbox.min.y) as f32,
                            1.0,
                            1.0,
                            &Source::Solid(SolidSource {
                                r: 0xff,
                                g: 0xff,
                                b: 0xff,
                                a: 0xff,
                            }),
                            &DrawOptions {
                                alpha: v,
                                ..DrawOptions::default()
                            },
                        );
                    });
                }
            }
        }

        let overlay: show_image::Image = dt.into();

        let img = img.clone();
        let window = show_image::context().run_function_wait(move |context| -> Result<_> {
            let mut window = context
                .create_window(
                    "vis",
                    WindowOptions {
                        size: Some([width, height]),
                        ..WindowOptions::default()
                    },
                )
                .context("Failed to create window")?;
            window.set_image(
                "picture",
                &img.as_image_view()
                    .context("Failed to image view of original image")?,
            );
            window.set_overlay(
                "yolo",
                &overlay
                    .as_image_view()
                    .context("Failed to set image view of overlay")?,
                true,
            );
            Ok(window.proxy())
        })?;

        for event in window.event_channel().unwrap() {
            if let WindowEvent::KeyboardInput(event) = event {
                if event.input.key_code == Some(VirtualKeyCode::Escape)
                    && event.input.state.is_pressed()
                {
                    break;
                }
            }
        }

        Ok(())
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
                .run(
                    inputs!["images" => input_tensor.view()]
                        .context("Failed to construct inputs")?,
                )
                .context("Failed to run model")?;
            let output = outputs["output0"]
                .try_extract_tensor::<f32>()
                .context("Failed to extract tensor from output")?
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

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
        let i = (h * 6.0).floor() as i32;
        let f = h * 6.0 - i as f32;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);

        let (r, g, b) = match i % 6 {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            5 => (v, p, q),
            _ => (1.0, 1.0, 1.0), // 默认白色
        };

        ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }

    fn get_color_for_class(class_id: usize) -> SolidSource {
        let num_classes = 84;
        let hue = (class_id as f32 / num_classes as f32) % 1.0;
        let (r, g, b) = Self::hsv_to_rgb(hue, 0.7, 0.9);

        SolidSource {
            r,
            g,
            b,
            a: 0xFF, // 不透明
        }
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

        let yolo = Yolo::new(0.0, 0.0, (2, 2), "");
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
        let yolo = Yolo::new(0.0, nms_threshold, (0, 0), "");
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
        let yolo = Yolo::new(conf_threshold, nms_threshold, (1, 1), "");

        let detections = yolo.process_yolov8_output(mock_output, (1, 1));

        assert_eq!(detections.len(), 2, "Incorrect size of detections");

        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[1].class_id, 1);

        assert!(detections[0].bbox.x_center > 0.5);
        assert!(detections[1].bbox.x_center < 0.5);
    }

    #[test]
    fn test_yolo() -> Result<()> {
        let yolo =
            Yolo::new(0.5, 0.75, (640, 640), "assets/test/yolov8n.onnx").build(Execution::CPU)?;

        let img = image::open(PathBuf::from_str("assets/test/zidane.jpg")?)?;
        let detections = yolo.infer(&img)?;

        assert!(detections.len() > 0);

        Ok(())
    }
}
