use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};
use log::trace;
use ndarray::{s, Array2, Array4};
use std::{error::Error, path::Path};

pub fn preprocess_image(
    image_path: &str,
    size: u32,
) -> Result<(Array4<f32>, (u32, u32)), Box<dyn Error>> {
    // 读取图像
    trace!("Reading image from path: {}", image_path);
    let img = image::open(&Path::new(image_path))?;

    // 调整图像大小
    trace!("Resizing image to {}x{}", size, size);
    let resized_img = img.resize_exact(size, size, FilterType::Nearest);

    // 如果不是 RGB 格式，则将图像转为 RGB 格式
    let rgb_image: RgbImage = match resized_img {
        DynamicImage::ImageRgb8(img) => {
            trace!("Image is already in RGB format");
            img
        }
        other => {
            trace!("Image is not RGB format, converting...");
            other.to_rgb8()
        }
    };

    // 归一化
    trace!("Normalizing the image pixels to range [0, 1]");
    let norm_img: Vec<f32> = rgb_image
        .pixels()
        .flat_map(|pixel| pixel.0.iter().map(|&p| p as f32 / 255.0))
        .collect();

    trace!(
        "Converting image data to 4D array with shape (1, 3, {}, {})",
        size,
        size
    );
    let array = Array4::from_shape_vec((1, 3, size as usize, size as usize), norm_img)?;

    Ok((array, img.dimensions()))
}

#[derive(Debug, Clone)]
pub struct BBox {
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    bbox: BBox,
    confidence: f32,
    class_id: u32,
}

fn non_max_suppression(detections: Vec<Detection>, nms_threshold: f32) -> Vec<Detection> {
    let mut detections = detections;
    let mut final_detections = Vec::new();

    // 按置信度降序排列
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    while !detections.is_empty() {
        // 选择置信度最高的框并移除
        let best_detection = detections.remove(0);
        final_detections.push(best_detection.clone());

        // 滤掉与当前最高置信度框 IoU 大于阈值的框
        detections.retain(|detection| {
            let iou = compute_iou(&best_detection.bbox, &detection.bbox);
            iou < nms_threshold
        });
    }

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

    let inter_area = (inter_x_max - inter_x_min).max(0.0) * (inter_y_max - inter_y_min).max(0.0);
    let bbox1_area = (x1_max - x1_min) * (y1_max - y1_min);
    let bbox2_area = (x2_max - x2_min) * (y2_max - y2_min);

    inter_area / (bbox1_area + bbox2_area - inter_area)
}

pub fn process_yolov8_output(
    output: Array2<f32>,
    conf_threshold: f32,
    nms_threshold: f32,
) -> Vec<Detection> {
    let num_predictions = output.shape()[0];

    let mut detections = Vec::new();

    // 解析输出张量
    for i in 0..num_predictions {
        let bbox = &output.slice(s![i, 0..4]); // [x_center, y_center, width, height]
        let class_scores = &output.slice(s![i, 4..]);

        // 获取最高类别置信度及其索引
        let (best_class_index, best_class_conf) = class_scores
            .iter()
            .enumerate()
            .map(|(index, &score)| (index, score))
            .max_by(|(_, conf1), (_, conf2)| conf1.partial_cmp(conf2).unwrap())
            .unwrap();

        // 过滤掉低置信度的框
        if best_class_conf > conf_threshold {
            let detection = Detection {
                bbox: BBox {
                    x_center: bbox[0],
                    y_center: bbox[1],
                    width: bbox[2],
                    height: bbox[3],
                },
                confidence: best_class_conf,
                class_id: best_class_index as u32,
            };
            detections.push(detection);
        }
    }

    // 应用非极大值抑制（NMS）
    let final_detections = non_max_suppression(detections, nms_threshold);

    final_detections
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use tempfile::tempdir;

    #[test]
    fn test_preprocess_image() -> Result<(), Box<dyn Error>> {
        // 准备：生成一个 4x4 的测试 RGB 图像
        let img = RgbImage::from_fn(4, 4, |x, y| {
            if (x + y) % 2 == 0 {
                image::Rgb([255, 0, 0]) // 红色
            } else {
                image::Rgb([0, 255, 0]) // 绿色
            }
        });

        // 将图片保存到临时路径
        let temp_dir = tempdir()?;
        let img_path = temp_dir.path().join("test_image.png");
        img.save(&img_path)?;

        // 调用函数，将图像调整到 2x2 大小
        let (processed_array, original_dims) = preprocess_image(
            img_path
                .to_str()
                .ok_or_else(|| format!("Failed to convert {:?} to string", img_path))?,
            2,
        )?;

        // 检查原始图像尺寸
        assert_eq!(original_dims, (4, 4));

        // 检查处理后的图像维度
        let shape = processed_array.shape();
        assert_eq!(shape, &[1, 3, 2, 2]); // batch_size=1, channels=3, height=2, width=2

        // 检查像素
        let processed_data = processed_array
            .as_slice()
            .ok_or_else(|| format!("Failed to convert {:?} to slice", processed_array))?;

        for &val in processed_data {
            assert!(val >= 0.0 && val <= 1.0, "Pixel value out of range: {val}");
        }

        Ok(())
    }

    #[test]
    fn test_non_max_suppression() {
        // 创建模拟检测框
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

        // 执行非极大值抑制
        let nms_threshold = 0.3;
        let final_detections = non_max_suppression(detections, nms_threshold);

        // 非极大值抑制后应该只剩下两个框
        assert_eq!(
            final_detections.len(),
            2,
            "Incorrect length of final detections"
        );

        // 验证剩下的框
        assert!(
            final_detections[0].confidence > final_detections[1].confidence,
            "Incorrect confidence sort"
        );
    }

    fn create_mock_yolov8_output() -> Array2<f32> {
        let num_predictions = 5;
        let num_classes = 80;
        let mut output = Array2::zeros((num_predictions, num_predictions + num_classes)); // [5, 85] 张量

        // 模拟边界框
        output[[0, 0]] = 0.7; // x_center
        output[[0, 1]] = 0.5; // y_center
        output[[0, 2]] = 0.2; // width
        output[[0, 3]] = 0.2; // height

        output[[1, 0]] = 0.3; // x_center
        output[[1, 1]] = 0.6; // y_center
        output[[1, 2]] = 0.2; // width
        output[[1, 3]] = 0.2; // height

        // 模拟类别置信度，假设类别0有较高置信度
        output[[0, 4]] = 0.9; // 类别0置信度
        output[[1, 5]] = 0.8;

        output
    }

    #[test]
    fn test_process_yolov8_output() {
        // 初始化模拟的 YOLOv8 输出
        let mock_output = create_mock_yolov8_output();

        // 设置置信度阈值和 NMS 阈值
        let conf_threshold = 0.5;
        let nms_threshold = 0.4;

        // 调用处理函数
        let detections = process_yolov8_output(mock_output, conf_threshold, nms_threshold);

        // 验证检测结果
        assert_eq!(detections.len(), 2, "Incorrect size of detections");

        // 验证检测的类别是否符合预期
        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[1].class_id, 1);

        // 验证检测框的坐标是否符合预期
        assert!(detections[0].bbox.x_center > 0.5);
        assert!(detections[1].bbox.x_center < 0.5);
    }
}
