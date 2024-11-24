use std::collections::{HashMap, VecDeque};

use anyhow::{anyhow, Result};
use image::{ImageBuffer, Luma};
use nalgebra::{Matrix3, Matrix4, Point3, Vector3, Vector4};
use rayon::prelude::*;
use tracing::{debug, error, span, trace, Level};

use super::detect::{BBox, RobotDetection};
use cluster::dbscan;

mod cluster;

#[derive(Debug)]
pub struct RobotLocation {
    pub center: Point3<f32>,
    pub width: f32,
    pub height: f32,
    pub depth: f32,
}

pub struct Locator {
    cluster_epsilon: f32,
    cluster_min_points: usize,
    min_valid_distance: f32,
    max_valid_distance: f32,
    background_depth_map: ImageBuffer<Luma<f32>, Vec<f32>>,
    depth_map_queue: VecDeque<ImageBuffer<Luma<f32>, Vec<f32>>>,
    lidar_to_camera_transform: Matrix4<f32>,
    camera_to_lidar_transform: Matrix4<f32>,
    camera_intrinsic: Matrix3<f32>,
    camera_intrinsic_inverse: Matrix3<f32>,
    max_depth_map_queue_size: usize,
}

impl Locator {
    pub fn new(
        cluster_epsilon: f32,
        cluster_min_points: usize,
        min_valid_distance: f32,
        max_valid_distance: f32,
        lidar_to_camera_transform: Matrix4<f32>,
        camera_intrinsic: Matrix3<f32>,
        max_depth_map_queue_size: usize,
    ) -> Result<Self> {
        let span = span!(Level::TRACE, "Locator::new");
        let _enter = span.enter();

        debug!(
            "Initializing Locator with cluster epsilon: {}, valid distance: {}~{}",
            cluster_epsilon, min_valid_distance, max_valid_distance,
        );

        trace!("Calculating transform matrix");
        let camera_to_lidar_transform = lidar_to_camera_transform
            .try_inverse()
            .ok_or_else(|| {
                anyhow!(
                    "Failed to invert lidar to camera transform: {:#?}",
                    lidar_to_camera_transform
                )
            })
            .map_err(|e| {
                error!(
                    "Failed to invert lidar to camera transform {:#?}: {}",
                    lidar_to_camera_transform, e
                );
                e
            })?;
        let camera_intrinsic_inverse = camera_intrinsic
            .try_inverse()
            .ok_or_else(|| anyhow!("Failed to invert camera intrinsic {:#?}", camera_intrinsic))?;
        debug!(
            "Camera-lidar transform: {}, camera intrinsic inverse: {}",
            camera_to_lidar_transform, camera_intrinsic_inverse
        );

        let locator = Self {
            cluster_epsilon,
            cluster_min_points,
            min_valid_distance,
            max_valid_distance,
            background_depth_map: ImageBuffer::default(),
            depth_map_queue: VecDeque::with_capacity(max_depth_map_queue_size),
            lidar_to_camera_transform,
            camera_to_lidar_transform,
            camera_intrinsic,
            camera_intrinsic_inverse,
            max_depth_map_queue_size,
        };
        Ok(locator)
    }

    pub fn locate_detections(
        &mut self,
        points: &[Point3<f32>],
        detections: &[RobotDetection],
    ) -> Result<Vec<Option<RobotLocation>>> {
        let span = span!(Level::TRACE, "Locator::locate_detections");
        let _enter = span.enter();

        debug!(
            "Locating detections with {} points and {} detections",
            points.len(),
            detections.len()
        );

        if self.background_depth_map.is_empty() {
            return Err(anyhow!("Background depth map is empty"));
        }

        trace!("Getting robot depth map");
        let robot_depth_map = self.get_robot_depth_map(points);

        trace!("Clustering and mapping categories.");
        let pixels_category_mapping = self.cluster_and_get_category(&robot_depth_map);

        trace!("Searching for robot location");
        let bboxes: Vec<_> = detections.iter().map(|det| det.bbox()).collect();
        let robot_locations =
            self.search_for_location(&bboxes, robot_depth_map, pixels_category_mapping);

        debug!("Robot locations found: {:?}", robot_locations);
        Ok(robot_locations)
    }

    pub fn update_background_depth_map(
        &mut self,
        points: &[Point3<f32>],
        depth_map_size: (u32, u32),
    ) -> Result<()> {
        let (image_width, image_height) = depth_map_size;
        if self.background_depth_map.is_empty() {
            debug!("Background depth map is empty, initializing.");
            self.background_depth_map = ImageBuffer::new(image_width, image_height);
        } else if self.background_depth_map.dimensions() != (image_width, image_height) {
            return Err(anyhow!(
                "Dimensions of image is not equal to background depth image"
            ));
        }

        let (image_width, image_height) = self.background_depth_map.dimensions();
        debug!("Image width and height: {image_width}x{image_height}");

        let image_points_filtered: Vec<_> = points
            .iter()
            .filter_map(|lidar_point| {
                if !lidar_point.is_empty()
                    && lidar_point.x.is_normal()
                    && lidar_point.y.is_normal()
                    && lidar_point.z.is_normal()
                    && lidar_point.x < self.max_valid_distance
                    && lidar_point.x > self.min_valid_distance
                {
                    let image_point = self.lidar_to_image(lidar_point);
                    debug!(
                        "Lidar point: {:?}, image point: {:?}",
                        lidar_point, image_point
                    );
                    let (u, v) = (image_point.x.round() as i32, image_point.y.round() as i32);
                    if u >= 0 && (u as u32) < image_width && v >= 0 && (v as u32) < image_height {
                        trace!("Point is in image dimension and kept.");
                        Some((u as u32, v as u32, image_point.z))
                    } else {
                        trace!("Point is out of image dimension and filtered.");
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        image_points_filtered.into_iter().for_each(|point| {
            let (u, v, depth) = point;
            let background_depth = self.background_depth_map.get_pixel_mut(u, v);
            if depth > background_depth.0[0] {
                background_depth.0[0] = depth;
            }
        });

        Ok(())
    }

    fn image_to_lidar(&self, point: &Point3<f32>) -> Point3<f32> {
        let camera_coor_vector = Vector3::new(point.x, point.y, 1.0);

        let (camera_to_lidar_rotate, camera_to_lidar_translate) = (
            self.camera_to_lidar_transform.fixed_view::<3, 3>(0, 0),
            self.camera_to_lidar_transform.fixed_view::<3, 1>(0, 3),
        );

        let lidar_coor_vector = camera_to_lidar_rotate
            * (self.camera_intrinsic_inverse * point.z * camera_coor_vector
                + camera_to_lidar_translate);
        Point3::new(
            lidar_coor_vector[0],
            lidar_coor_vector[1],
            lidar_coor_vector[2],
        )
    }

    fn lidar_to_image(&self, point: &Point3<f32>) -> Point3<f32> {
        let lidar_coor_vector = Vector4::new(point.x, point.y, point.z, 1.0);

        let camera_coor_vector = self.camera_intrinsic
            * (self.lidar_to_camera_transform * lidar_coor_vector).view((0, 0), (3, 1));
        Point3::new(
            camera_coor_vector[0] / camera_coor_vector[2],
            camera_coor_vector[1] / camera_coor_vector[2],
            camera_coor_vector[2],
        )
    }

    fn get_robot_depth_map(&mut self, points: &[Point3<f32>]) -> ImageBuffer<Luma<f32>, Vec<f32>> {
        let span = span!(Level::TRACE, "Locator::get_robot_depth_map");
        let _enter = span.enter();

        let (image_width, image_height) = self.background_depth_map.dimensions();
        debug!("Image width and height: {image_width}x{image_height}");

        trace!("Generating robot depth map with {} points", points.len());
        let image_points_filtered: Vec<_> = points
            .iter()
            .filter_map(|lidar_point| {
                if !lidar_point.is_empty()
                    && lidar_point.x.is_normal()
                    && lidar_point.y.is_normal()
                    && lidar_point.z.is_normal()
                    && lidar_point.x < self.max_valid_distance
                    && lidar_point.x > self.min_valid_distance
                {
                    let image_point = self.lidar_to_image(lidar_point);
                    debug!(
                        "Lidar point: {:?}, image point: {:?}",
                        lidar_point, image_point
                    );
                    let (u, v) = (image_point.x.round() as i32, image_point.y.round() as i32);
                    if u >= 0 && (u as u32) < image_width && v >= 0 && (v as u32) < image_height {
                        trace!("Point is in image dimension and kept.");
                        Some((u as u32, v as u32, image_point.z))
                    } else {
                        trace!("Point is out of image dimension and filtered.");
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        debug!(
            "Filtered image point number: {}",
            image_points_filtered.len()
        );

        let mut depth_map: ImageBuffer<Luma<f32>, Vec<_>> =
            ImageBuffer::new(image_width, image_height);
        image_points_filtered.into_iter().for_each(|point| {
            let (u, v, depth) = point;
            depth_map.put_pixel(u, v, Luma([depth]));
        });
        self.depth_map_queue.push_back(depth_map);
        if self.depth_map_queue.len() > self.max_depth_map_queue_size {
            self.depth_map_queue.pop_front();
        }

        let mut difference_depth_map: ImageBuffer<Luma<f32>, Vec<f32>> =
            ImageBuffer::new(image_width, image_height);
        self.depth_map_queue.iter().for_each(|depth_map| {
            difference_depth_map
                .enumerate_pixels_mut()
                .par_bridge()
                .for_each(|(x, y, pixel)| {
                    let depth_value = depth_map.get_pixel(x, y).0[0];
                    let background_depth_value = self.background_depth_map.get_pixel(x, y).0[0];

                    let difference = (depth_value - background_depth_value).abs();
                    if difference > self.min_valid_distance && difference < self.max_valid_distance
                    {
                        debug!("Difference depth map x: {}, y: {}, depth: {}, background: {}, difference: {}", x, y, depth_value, background_depth_value, difference);
                        pixel.0[0] = difference;
                    }
                });
        });

        difference_depth_map
    }

    fn cluster_and_get_category(
        &self,
        difference_depth_map: &ImageBuffer<Luma<f32>, Vec<f32>>,
    ) -> HashMap<(u32, u32), isize> {
        let span = span!(Level::TRACE, "Locator::cluster_and_get_category");
        let _enter = span.enter();

        debug!("Clustering depth map into categories");
        let image_points: Vec<_> = difference_depth_map
            .enumerate_pixels()
            .par_bridge()
            .filter_map(|(x, y, pixel)| {
                let depth = pixel.0[0];
                if depth.is_normal() {
                    Some((x, y, depth))
                } else {
                    None
                }
            })
            .collect();

        let lidar_points: Vec<Point3<f32>> = image_points
            .iter()
            .map(|(x, y, depth)| {
                let image_point = Point3::new(*x as f32, *y as f32, *depth);

                let lidar_point = self.image_to_lidar(&image_point);

                debug!(
                    "Image point: {:?}, lidar point: {:?}",
                    image_point, lidar_point
                );

                lidar_point
            })
            .collect();

        let categories = dbscan(&lidar_points, self.cluster_epsilon, self.cluster_min_points);

        let mut mapping = HashMap::with_capacity(categories.len());
        image_points
            .into_iter()
            .zip(categories.into_iter())
            .for_each(|((pixel_x, pixel_y, _depth), category)| {
                mapping.insert((pixel_x, pixel_y), category);
            });

        mapping
    }

    fn search_for_location(
        &self,
        bboxes: &[BBox],
        difference_depth_map: ImageBuffer<Luma<f32>, Vec<f32>>,
        cluster_result: HashMap<(u32, u32), isize>,
    ) -> Vec<Option<RobotLocation>> {
        let span = span!(Level::TRACE, "Locator::search_for_location");
        let _enter = span.enter();

        debug!(
            "Searching for robot locations in {} bounding boxes",
            bboxes.len()
        );
        let (image_width, image_height) = difference_depth_map.dimensions();

        bboxes
            .iter()
            .map(|bbox| {
                let mut category_pixels: HashMap<isize, Vec<(u32, u32)>> = HashMap::new();
                let (x_min, x_max, y_min, y_max) = (
                    (bbox.x_center - bbox.width / 2.0).max(0.0).floor() as u32,
                    (bbox.x_center + bbox.width / 2.0).ceil() as u32,
                    (bbox.y_center - bbox.height / 2.0).max(0.0).floor() as u32,
                    (bbox.y_center + bbox.height / 2.0).ceil() as u32,
                );

                for y in y_min..=y_max {
                    if y >= image_height {
                        break;
                    }
                    for x in x_min..=x_max {
                        if x >= image_width {
                            break;
                        }
                        if let Some(&category) = cluster_result.get(&(x, y)) {
                            category_pixels
                                .entry(category)
                                .or_insert_with(Vec::new)
                                .push((x, y));
                        }
                    }
                }

                if let Some((_, pixels)) = category_pixels
                    .iter()
                    .max_by_key(|&(_, pixels)| pixels.len())
                {
                    let (sum_point, count, min_max) = pixels
                        .iter()
                        .filter_map(|&(x, y)| {
                            let depth = difference_depth_map.get_pixel(x, y).0[0];
                            if depth.is_normal() {
                                let image_point = Point3::new(x as f32, y as f32, depth);
                                let lidar_point = self.image_to_lidar(&image_point);
                                Some(lidar_point)
                            } else {
                                None
                            }
                        })
                        .fold(
                            (
                                Point3::<f32>::new(0.0, 0.0, 0.0),
                                0,
                                (
                                    Point3::<f32>::new(f32::MAX, f32::MAX, f32::MAX),
                                    Point3::<f32>::new(f32::MIN, f32::MIN, f32::MIN),
                                ),
                            ),
                            |(sum, cnt, (min_point, max_point)), point| {
                                (
                                    Point3::new(sum.x + point.x, sum.y + point.y, sum.z + point.z),
                                    cnt + 1,
                                    (
                                        Point3::new(
                                            min_point.x.min(point.x),
                                            min_point.y.min(point.y),
                                            min_point.z.min(point.z),
                                        ),
                                        Point3::new(
                                            max_point.x.max(point.x),
                                            max_point.y.max(point.y),
                                            max_point.z.max(point.z),
                                        ),
                                    ),
                                )
                            },
                        );

                    if count > 0 {
                        let robot_location = RobotLocation {
                            center: Point3::new(
                                sum_point.x / count as f32,
                                sum_point.y / count as f32,
                                sum_point.z / count as f32,
                            ),
                            width: min_max.1.y - min_max.0.y,
                            height: min_max.1.z - min_max.0.z,
                            depth: min_max.1.x - min_max.0.x,
                        };

                        Some(robot_location)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::{Matrix3, Matrix4, Point3};

    #[test]
    fn test_lidar_image_conversion() {
        let camera_intrinsic = Matrix3::<f32>::identity();
        let lidar_to_camera_transform = Matrix4::<f32>::identity();

        let locator = Locator::new(
            0.5,
            10,
            0.1,
            100.0,
            lidar_to_camera_transform,
            camera_intrinsic,
            4,
        )
        .unwrap();

        let lidar_point = Point3::new(1.0, 2.0, 3.0);
        let image_point = locator.lidar_to_image(&lidar_point);
        let converted_back = locator.image_to_lidar(&image_point);

        assert_approx_eq!((lidar_point - converted_back).norm(), 0.0);
    }

    #[test]
    fn test_get_robot_depth_map() {
        let camera_intrinsic = Matrix3::<f32>::identity();
        let lidar_to_camera_transform = Matrix4::<f32>::identity();

        let mut locator = Locator::new(
            0.5,
            10,
            0.1,
            100.0,
            lidar_to_camera_transform,
            camera_intrinsic,
            4,
        )
        .unwrap();
        locator.background_depth_map = ImageBuffer::new(10, 10);

        let mut points = vec![Point3::new(4.0, 6.0, 2.0)];

        locator.get_robot_depth_map(&points);

        points = vec![Point3::new(2.0, 3.0, 1.0)];
        let depth_map = locator.get_robot_depth_map(&points);

        let pixel = depth_map.get_pixel(2, 3);
        assert_approx_eq!(pixel.0[0], 1.0);
    }
}
