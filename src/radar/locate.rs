use std::collections::HashMap;

use anyhow::{anyhow, Result};
use dbscan::Classification;
use image::{ImageBuffer, Luma};
use nalgebra::{Matrix3, Matrix4, Point3, Vector3, Vector4};
use rayon::prelude::*;
use tracing::{debug, error, span, trace, Level};

use super::detect::{BBox, RobotDetection};
use crate::config::{LocatorConfig, RadarInstanceConfig};

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
    min_valid_distance_diff: f32,
    max_valid_distance_diff: f32,
    scale_factor: f32,
    zoom_factor: f32,
    roi_offset: (u32, u32),
    background_depth_map: ImageBuffer<Luma<f32>, Vec<f32>>,
    lidar_to_camera_transform: Matrix4<f32>,
    camera_to_lidar_transform: Matrix4<f32>,
    camera_intrinsic: Matrix3<f32>,
    camera_intrinsic_inverse: Matrix3<f32>,
}

impl Locator {
    pub fn new(
        cluster_epsilon: f32,
        cluster_min_points: usize,
        min_valid_distance: f32,
        max_valid_distance: f32,
        min_valid_distance_diff: f32,
        max_valid_distance_diff: f32,
        scale_factor: f32,
        zoom_factor: f32,
        roi_offset: (u32, u32),
        lidar_to_camera_transform: Matrix4<f32>,
        camera_intrinsic: Matrix3<f32>,
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
                anyhow!(
                    "Failed to invert lidar to camera transform {:#?}: {}",
                    lidar_to_camera_transform,
                    e
                )
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
            min_valid_distance_diff,
            max_valid_distance_diff,
            scale_factor,
            zoom_factor,
            roi_offset,
            background_depth_map: ImageBuffer::default(),
            lidar_to_camera_transform,
            camera_to_lidar_transform,
            camera_intrinsic,
            camera_intrinsic_inverse,
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

        if detections.is_empty() {
            trace!("Detections is empty, no need for location.");
            return Ok(Vec::new());
        }

        trace!("Getting robot depth map");
        let robot_depth_map = self.get_robot_depth_map(points);

        trace!("Searching for robot location");
        let bboxes: Vec<_> = detections.iter().map(|det| det.bbox()).collect();
        let robot_locations = self.search_for_location(&bboxes, &robot_depth_map);

        debug!("Robot locations found: {:?}", robot_locations);
        Ok(robot_locations)
    }

    pub fn from_config(
        locator_config: &LocatorConfig,
        instance_config: &RadarInstanceConfig,
    ) -> Result<Self> {
        Locator::new(
            locator_config.cluster_epsilon,
            locator_config.cluster_min_points,
            locator_config.min_valid_distance,
            locator_config.max_valid_distance,
            locator_config.min_valid_distance_diff,
            locator_config.max_valid_distance_diff,
            locator_config.scale_factor,
            locator_config.zoom_factor,
            instance_config.roi_offset.into(),
            Matrix4::from_row_slice(&instance_config.lidar_to_camera),
            Matrix3::from_row_slice(&instance_config.intrinsic),
        )
    }

    pub fn update_background_depth_map(
        &mut self,
        points: &[Point3<f32>],
        image_size: (u32, u32),
    ) -> Result<()> {
        let (image_width, image_height) = image_size;

        let depth_map_width = (image_width as f32 * self.zoom_factor) as u32;
        let depth_map_height = (image_height as f32 * self.zoom_factor) as u32;

        if self.background_depth_map.is_empty() {
            debug!("Background depth map is empty, initializing.");
            self.background_depth_map = ImageBuffer::new(depth_map_width, depth_map_height);
        } else if self.background_depth_map.dimensions() != (depth_map_width, depth_map_height) {
            return Err(anyhow!(
                "Dimensions of image is not equal to background depth image"
            ));
        }

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
                    let (u, v) = (image_point.x.round() as i32, image_point.y.round() as i32);
                    if u >= 0
                        && (u as u32) < depth_map_width
                        && v >= 0
                        && (v as u32) < depth_map_height
                    {
                        Some((u as u32, v as u32, image_point.z))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        debug!(
            "Size of filtered points in updating background depth map: {}",
            image_points_filtered.len()
        );

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
        let camera_coor_vector =
            Vector3::new(point.x / self.zoom_factor, point.y / self.zoom_factor, 1.0);

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
            camera_coor_vector[0] * self.zoom_factor / camera_coor_vector[2],
            camera_coor_vector[1] * self.zoom_factor / camera_coor_vector[2],
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
                    let (u, v) = (image_point.x.round() as i32, image_point.y.round() as i32);
                    if u >= 0 && (u as u32) < image_width && v >= 0 && (v as u32) < image_height {
                        Some((u as u32, v as u32, image_point.z))
                    } else {
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

        let mut difference_depth_map: ImageBuffer<Luma<f32>, Vec<f32>> =
            ImageBuffer::new(image_width, image_height);

        difference_depth_map
            .iter_mut()
            .zip(depth_map.into_iter())
            .zip(self.background_depth_map.iter())
            .par_bridge()
            .for_each(|((diff_depth, depth), background_depth)| {
                if !depth.is_normal() || !background_depth.is_normal() {
                    return;
                }
                let difference = background_depth - depth;
                if difference > self.min_valid_distance_diff
                    && difference < self.max_valid_distance_diff
                {
                    *diff_depth = difference;
                }
            });

        difference_depth_map
    }

    fn search_for_location(
        &self,
        bboxes: &[BBox],
        difference_depth_map: &ImageBuffer<Luma<f32>, Vec<f32>>,
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
                let x_min =
                    (bbox.x_center - bbox.width * self.scale_factor * 0.5) * self.zoom_factor;
                let x_max =
                    (bbox.x_center + bbox.width * self.scale_factor * 0.5) * self.zoom_factor;
                let y_min =
                    (bbox.y_center - bbox.height * self.scale_factor * 0.5) * self.zoom_factor;
                let y_max =
                    (bbox.y_center + bbox.height * self.scale_factor * 0.5) * self.zoom_factor;

                let x_min = (x_min + self.roi_offset.0 as f32).max(0.0) as u32;
                let x_max = (x_max + self.roi_offset.0 as f32).min(image_width as f32) as u32;
                let y_min = (y_min + self.roi_offset.1 as f32).max(0.0) as u32;
                let y_max = (y_max + self.roi_offset.1 as f32).min(image_height as f32) as u32;

                debug!("BBox: ({x_min}, {y_min})~({x_max}, {y_max})");

                let size = ((y_max - y_min) * (x_max - x_min)) as usize;
                let mut image_points = Vec::with_capacity(size);
                let mut lidar_points = Vec::with_capacity(size);
                for y in y_min..y_max {
                    for x in x_min..x_max {
                        let depth = difference_depth_map.get_pixel(x, y).0[0];
                        if depth > self.min_valid_distance && depth < self.max_valid_distance {
                            debug!("x: {x}, y: {y}, depth: {depth}");
                            let image_point = Point3::new(x as f32, y as f32, depth);
                            let lidar_point = self.image_to_lidar(&image_point);
                            lidar_points.push(vec![lidar_point.x, lidar_point.y, lidar_point.z]);
                            image_points.push(image_point);
                        }
                    }
                }

                let dbscan_model =
                    dbscan::Model::new(self.cluster_epsilon as f64, self.cluster_min_points);
                let categories = dbscan_model.run(&lidar_points);

                let mut category_mapping: HashMap<usize, Vec<Point3<f32>>> =
                    HashMap::with_capacity(size);
                image_points.iter().zip(categories.into_iter()).for_each(
                    |(image_point, category)| {
                        if let Classification::Core(category) = category {
                            category_mapping
                                .entry(category)
                                .or_insert_with(Vec::new)
                                .push(*image_point);
                        }
                    },
                );
                debug!("Category of pixels: {:?}", category_mapping);

                let pixels = if let Some((category, pixels)) = category_mapping
                    .iter()
                    .max_by_key(|&(_, pixels)| pixels.len())
                {
                    trace!("Category {category} selected for location");
                    pixels
                } else {
                    trace!("No category selected, will return average of points");
                    &image_points
                };

                debug!("Pixels: {:?}", pixels);

                if pixels.is_empty() {
                    return None;
                }

                let (sum_point, count, min_point, max_point) = pixels
                    .iter()
                    .filter_map(|image_point| {
                        if image_point.z.is_normal() {
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
                            Point3::<f32>::new(f32::MAX, f32::MAX, f32::MAX),
                            Point3::<f32>::new(f32::MIN, f32::MIN, f32::MIN),
                        ),
                        |(sum, cnt, min_point, max_point), point| {
                            (
                                Point3::new(sum.x + point.x, sum.y + point.y, sum.z + point.z),
                                cnt + 1,
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
                            )
                        },
                    );

                let robot_location = RobotLocation {
                    center: Point3::new(
                        sum_point.x / count as f32,
                        sum_point.y / count as f32,
                        sum_point.z / count as f32,
                    ),
                    width: max_point.y - min_point.y,
                    height: max_point.z - min_point.z,
                    depth: max_point.x - min_point.x,
                };

                debug!("robot location is {:?}", robot_location);
                Some(robot_location)
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
            0.1,
            100.0,
            1.0,
            1.0,
            (0, 0),
            lidar_to_camera_transform,
            camera_intrinsic,
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
            0.1,
            100.0,
            1.0,
            1.0,
            (0, 0),
            lidar_to_camera_transform,
            camera_intrinsic,
        )
        .unwrap();
        locator.background_depth_map = ImageBuffer::new(10, 10);

        let mut points = vec![Point3::new(4.0, 6.0, 2.0)];

        locator
            .update_background_depth_map(&points, (10, 10))
            .unwrap();

        points = vec![Point3::new(2.0, 3.0, 1.0)];
        let depth_map = locator.get_robot_depth_map(&points);

        let pixel = depth_map.get_pixel(2, 3);
        assert_approx_eq!(pixel.0[0], 1.0);
    }
}
