mod config;
pub mod detect;
mod locate;

use anyhow::{anyhow, Context, Result};
use config::RadarConfig;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use locate::{Locator, RobotLocation};
use nalgebra::{Matrix3, Matrix4, Point3, Vector5};
use opencv::{
    calib3d,
    core::{Mat, CV_32F, CV_8UC3},
    prelude::*,
};

use detect::{Execution, RobotDetection, RobotDetector};
use tracing::{debug, error, span, trace, Level};

#[derive(Debug)]
pub struct RobotInfo {
    detection: RobotDetection,
    location: RobotLocation,
}

pub struct RadarInstanceConfig {
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
        instance_config: &RadarInstanceConfig,
    ) -> Result<Vec<RobotInfo>> {
        let span = span!(Level::TRACE, "Radar::detect_and_locate");
        let _enter = span.enter();

        trace!("Undistorting image...");
        let image_undistorted = Self::undistort_dynamic_image(
            image,
            &instance_config.camera_intrinsic,
            &instance_config.distortion,
        )
        .context("Failed to undistort image")?;

        trace!("Running detection and location...");
        let detect_result = self.robot_detector.detect(&image_undistorted)?;
        let locate_result = self.locator.locate_detections(
            &point_cloud,
            &detect_result,
            &image_undistorted,
            &instance_config.lidar_to_world_transform,
            &instance_config.lidar_to_camera_transform,
            &instance_config.camera_intrinsic,
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

    fn dynamic_image_to_mat(img: &DynamicImage) -> Result<Mat> {
        let span = span!(Level::TRACE, "Radar::dynamic_image_to_mat");
        let _enter = span.enter();

        let (width, height) = img.dimensions();
        debug!("Image dimensions: width = {}, height = {}", width, height);

        if let DynamicImage::ImageRgb8(ref rgb_img) = img {
            let flat_samples = rgb_img.as_flat_samples();
            let buffer = flat_samples.as_slice();
            debug!("DynamicImage data pointer: {:p}", buffer.as_ptr());
            debug!("DynamicImage buffer length: {}", buffer.len());

            let mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe_def(
                    height as i32,
                    width as i32,
                    CV_8UC3,
                    buffer.as_ptr() as *mut std::ffi::c_void,
                )
            }
            .context("Failed to construct mat with slice")?;

            trace!("Mat successfully constructed from DynamicImage.");
            Ok(mat)
        } else {
            let err_info = format!("Unupprorted image color {:?}.", img.color());
            error!(err_info);
            Err(anyhow!(err_info))
        }
    }

    fn mat_to_dynamic_image(mat: &impl MatTraitConst) -> Result<DynamicImage> {
        let span = span!(Level::TRACE, "Radar::mat_to_dynamic_image");
        let _enter = span.enter();

        let mat_size = mat.size().context("Failed to get Mat shape")?;
        let (width, height) = (mat_size.width as u32, mat_size.height as u32);
        debug!("Mat dimensions: width = {}, height = {}", width, height);

        let mat_data_ptr = mat.data();
        let length = mat.step1(0).context("Failed to get step of Mat")? * height as usize;
        debug!("Mat data pointer: {:p}", mat_data_ptr);
        debug!("Mat data length: {}", length);

        if mat.typ() == CV_8UC3 {
            trace!("Mat type is CV_8UC3, proceeding to create ImageBuffer.");

            let buf = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, unsafe {
                // TODO: 避免显式拷贝数据
                std::slice::from_raw_parts(mat_data_ptr, length).to_vec()
            })
            .ok_or_else(|| anyhow!("Failed to create ImageBuffer from mat"))?;

            trace!("ImageBuffer successfully created from Mat.");
            Ok(DynamicImage::ImageRgb8(buf))
        } else {
            let err_info = format!("Unsupported Mat type {}", mat.typ());
            error!(err_info);
            Err(anyhow!(err_info))
        }
    }

    fn matrix3_to_mat(matrix: &Matrix3<f32>) -> Result<Mat> {
        // OpenCV 内存序与 nalgrbra 相反，因此需要转置
        let mat = Mat::new_rows_cols_with_data(3, 3, matrix.as_slice())
            .context("Failed to create OpenCV Mat from Matrix3<f32>")?;

        let mut transposed_mat = Mat::default();
        opencv::core::transpose(&mat, &mut transposed_mat).context("Failed to transpose Mat")?;

        Ok(transposed_mat)
    }

    fn vector5_to_mat(vector: &Vector5<f32>) -> Result<Mat> {
        let data_ptr = vector.as_ptr() as *const f32;
        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe_def(5, 1, CV_32F, data_ptr as *mut std::ffi::c_void)
                .context("Failed to create OpenCV Mat from Vector5<f32>")?
        };

        Ok(mat)
    }

    pub fn undistort_dynamic_image(
        img: &DynamicImage,
        camera_matrix: &Matrix3<f32>,
        dist_coeffs: &Vector5<f32>,
    ) -> Result<DynamicImage> {
        let span = span!(Level::TRACE, "Radar::undistort_dynamic_image");
        let _enter = span.enter();

        trace!("Starting undistortion process");
        let camera_matrix = Self::matrix3_to_mat(camera_matrix)?;

        trace!("Converting distortion coefficients to OpenCV Mat");
        let dist_coeffs = Self::vector5_to_mat(dist_coeffs)?;

        trace!("Converting input DynamicImage to OpenCV Mat");
        let mat_img = Self::dynamic_image_to_mat(img)?;

        trace!("Performing undistortion on the image");
        let mut undistorted_img = Mat::default();
        calib3d::undistort(
            &mat_img,
            &mut undistorted_img,
            &camera_matrix,
            &dist_coeffs,
            &camera_matrix,
        )
        .context("Failed to undistort Mat")?;

        trace!("Converting undistorted Mat back to DynamicImage");
        let result_img = Self::mat_to_dynamic_image(&undistorted_img)?;

        trace!("Undistortion process completed");
        Ok(result_img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use image::RgbImage;
    use nalgebra::{Matrix3, Vector5};
    use opencv::core::{Vec3b, CV_32F};

    #[test]
    fn test_matrix3_to_mat() {
        let matrix = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let mat = Radar::matrix3_to_mat(&matrix).unwrap();

        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 3);
        assert_eq!(mat.typ(), CV_32F);

        for row in 0..3 {
            for col in 0..3 {
                let value: f32 = *mat.at_2d(row, col).unwrap();
                let expected_value = matrix[(row as usize, col as usize)];
                assert_approx_eq!(value, expected_value);
            }
        }
    }

    #[test]
    fn test_vector5_to_mat() {
        let vector = Vector5::new(1.0, 2.0, 3.0, 4.0, 5.0);

        let mat_result = Radar::vector5_to_mat(&vector);
        assert!(mat_result.is_ok());

        let mat = mat_result.unwrap();

        assert_eq!(mat.rows(), 5);
        assert_eq!(mat.cols(), 1);
        assert_eq!(mat.typ(), CV_32F);

        let mat_data: Vec<f32> = mat.data_typed().unwrap().to_vec();
        let expected_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(mat_data, expected_data);
    }

    fn create_test_dynamic_image() -> DynamicImage {
        let width = 3;
        let height = 2;
        let mut img = RgbImage::new(width, height);

        img.put_pixel(0, 0, Rgb([255, 0, 0])); // Red
        img.put_pixel(1, 0, Rgb([0, 255, 0])); // Green
        img.put_pixel(2, 0, Rgb([0, 0, 255])); // Blue
        img.put_pixel(0, 1, Rgb([255, 255, 0])); // Yellow
        img.put_pixel(1, 1, Rgb([0, 255, 255])); // Cyan
        img.put_pixel(2, 1, Rgb([255, 0, 255])); // Magenta

        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_dynamic_image_to_mat() {
        let dynamic_image = create_test_dynamic_image();

        let mat = Radar::dynamic_image_to_mat(&dynamic_image)
            .expect("Failed to convert DynamicImage to Mat");

        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 3);
        assert_eq!(mat.typ(), CV_8UC3);

        for y in 0..2 {
            for x in 0..3 {
                let pixel = mat
                    .at_2d::<Vec3b>(y, x)
                    .expect("Failed to access Mat pixel");
                let expected_pixel = dynamic_image.get_pixel(x as u32, y as u32);
                assert_eq!(pixel.0[0], expected_pixel.0[0]);
                assert_eq!(pixel.0[1], expected_pixel.0[1]);
                assert_eq!(pixel.0[2], expected_pixel.0[2]);
            }
        }
    }

    #[test]
    fn test_mat_to_dynamic_image() {
        let rows = 2;
        let cols = 3;
        let data = vec![
            Vec3b::from_array([255, 0, 0]),   // Red
            Vec3b::from_array([0, 255, 0]),   // Green
            Vec3b::from_array([0, 0, 255]),   // Blue
            Vec3b::from_array([255, 255, 0]), // Yellow
            Vec3b::from_array([0, 255, 255]), // Cyan
            Vec3b::from_array([255, 0, 255]), // Magenta
        ];

        let mat = Mat::new_rows_cols_with_data(rows, cols, &data).expect("Failed to create Mat");

        let dynamic_image =
            Radar::mat_to_dynamic_image(&mat).expect("Failed to convert Mat to DynamicImage");

        let (width, height) = dynamic_image.dimensions();
        assert_eq!(width, 3);
        assert_eq!(height, 2);

        if let DynamicImage::ImageRgb8(dynamic_image) = dynamic_image {
            for y in 0..2 {
                for x in 0..3 {
                    let pixel = mat
                        .at_2d::<Vec3b>(y, x)
                        .expect("Failed to access Mat pixel");
                    let expected_pixel = dynamic_image.get_pixel(x as u32, y as u32);
                    assert_eq!(pixel.0[0], expected_pixel.0[0]);
                    assert_eq!(pixel.0[1], expected_pixel.0[1]);
                    assert_eq!(pixel.0[2], expected_pixel.0[2]);
                }
            }
        } else {
            panic!("Unexpected DynamicImage type");
        }
    }
}
