use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use nalgebra::{ArrayStorage, Const, Matrix, Matrix3, Vector5};
use opencv::{
    calib3d::{get_optimal_new_camera_matrix, undistort},
    core::{transpose, Mat, Rect, Size, CV_8UC3},
    prelude::*,
};
use tracing::{debug, span, trace, Level};

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
        Err(anyhow!("Unupprorted image color {:?}.", img.color()))
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
        Err(anyhow!("Unsupported Mat type {}", mat.typ()))
    }
}

fn matrix_to_mat<T, const R: usize, const C: usize>(
    matrix: &Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>,
) -> Result<Mat>
where
    T: nalgebra::Scalar + opencv::prelude::DataType,
{
    let rows = matrix.nrows();
    let cols = matrix.ncols();

    let mat = Mat::new_rows_cols_with_data(cols as i32, rows as i32, matrix.as_slice())
        .context("Failed to create OpenCV Mat from nalgebra matrix")?;

    let mut transposed_mat = Mat::default();
    transpose(&mat, &mut transposed_mat).context("Failed to transpose Mat")?;

    Ok(transposed_mat)
}

pub fn undistort_image(
    img: &DynamicImage,
    camera_matrix: &mut Matrix3<f32>,
    dist_coeffs: &Vector5<f32>,
) -> Result<DynamicImage> {
    let span = span!(Level::TRACE, "Radar::undistort_dynamic_image");
    let _enter = span.enter();

    trace!("Starting undistortion process");
    let camera_matrix_mat = matrix_to_mat(camera_matrix)?;

    trace!("Converting distortion coefficients to OpenCV Mat");
    let dist_coeffs = matrix_to_mat(dist_coeffs)?;

    trace!("Converting input DynamicImage to OpenCV Mat");
    let mat_img = dynamic_image_to_mat(img)?;

    let img_size = Size {
        width: mat_img.cols(),
        height: mat_img.rows(),
    };
    let mut valid_roi = Rect::default();
    let new_camera_matrix_mat = get_optimal_new_camera_matrix(
        &camera_matrix_mat,
        &dist_coeffs,
        img_size,
        1.0,
        img_size,
        Some(&mut valid_roi),
        false,
    )?;

    trace!("Performing undistortion on the image");
    let mut undistorted_img = Mat::default();
    undistort(
        &mat_img,
        &mut undistorted_img,
        &camera_matrix_mat,
        &dist_coeffs,
        &new_camera_matrix_mat,
    )
    .context("Failed to undistort Mat")?;

    let cropped_img =
        Mat::roi(&undistorted_img, valid_roi).context("Failed to crop undistorted image")?;

    trace!("Converting undistorted Mat back to DynamicImage");
    let result_img = mat_to_dynamic_image(&cropped_img)?;

    trace!("Undistortion process completed");
    Ok(result_img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use image::RgbImage;
    use nalgebra::{Matrix3, Matrix4x3, Vector5};
    use opencv::core::Vec3b;

    #[test]
    fn test_matrix_to_mat() -> Result<()> {
        #[rustfmt::skip]
        let matrix = Matrix4x3::new(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        );

        let mat = matrix_to_mat(&matrix)?;

        for row in 0..4 {
            for col in 0..3 {
                let pixel_mat = mat.at_2d::<f64>(row, col)?;
                let pixel_matrix = matrix[(row as usize, col as usize)];
                assert_approx_eq!(pixel_mat, pixel_matrix);
            }
        }

        Ok(())
    }

    fn create_test_image() -> DynamicImage {
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
        let dynamic_image = create_test_image();

        let mat =
            dynamic_image_to_mat(&dynamic_image).expect("Failed to convert DynamicImage to Mat");

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
            mat_to_dynamic_image(&mat).expect("Failed to convert Mat to DynamicImage");

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

    fn create_camera_matrix() -> Matrix3<f32> {
        Matrix3::new(300.0, 0.0, 150.0, 0.0, 300.0, 150.0, 0.0, 0.0, 1.0)
    }

    fn create_dist_coeffs() -> Vector5<f32> {
        Vector5::new(0.1, 0.01, 0.001, 0.0, 0.0)
    }

    #[test]
    fn test_undistort_image() -> Result<()> {
        let test_img = create_test_image();

        let mut camera_matrix = create_camera_matrix();
        let dist_coeffs = create_dist_coeffs();

        let result_img = undistort_image(&test_img, &mut camera_matrix, &dist_coeffs)?;
        assert_eq!(result_img.dimensions(), (2, 1));

        if let DynamicImage::ImageRgb8(rgb_image) = result_img {
            let pixel_0 = rgb_image.get_pixel(0, 0);
            assert_eq!(pixel_0.0[0], 202);
            assert_eq!(pixel_0.0[1], 0);
            assert_eq!(pixel_0.0[2], 0);

            let pixel_0 = rgb_image.get_pixel(1, 0);
            assert_eq!(pixel_0.0[0], 0);
            assert_eq!(pixel_0.0[1], 127);
            assert_eq!(pixel_0.0[2], 112);
        } else {
            return Err(anyhow!("Unexpected image type"));
        }

        Ok(())
    }
}
