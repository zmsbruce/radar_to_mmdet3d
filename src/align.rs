use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{debug, error, info, trace, warn};

use crate::io::{hdf5::Hdf5PointCloudReader, pcd::write_points_to_pcd, video::VideoReader};

pub struct Aligner {
    video_readers: Vec<VideoReader>,
    point_cloud_reader: Hdf5PointCloudReader,
}

impl Aligner {
    pub fn new(video_file_paths: &[&str], pointcloud_file_path: &str) -> Result<Self> {
        let video_readers: Result<Vec<_>> = video_file_paths
            .iter()
            .map(|video_file_path| VideoReader::from_file(*video_file_path))
            .collect();

        let point_cloud_reader = Hdf5PointCloudReader::from_file(pointcloud_file_path)
            .context("Failed to construct point cloud reader")
            .map_err(|e| {
                error!("Failed to construct point cloud reader: {e}");
                e
            })?;

        Ok(Self {
            video_readers: video_readers
                .context("Failed to construct video readers")
                .map_err(|e| {
                    error!("Failed to construct video readers: {e}");
                    e
                })?,
            point_cloud_reader,
        })
    }

    fn get_align_frame_count(&self) -> Result<usize> {
        let min_video_frames = self
            .video_readers
            .iter()
            .enumerate()
            .map(|(idx, reader)| {
                let frames = reader
                    .total_frames()
                    .context("Failed to get total frames of video")?;

                info!("Total frames of video {}: {}", idx, frames);
                Ok(frames)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min()
            .ok_or_else(|| anyhow!("Total frames iterator is empty"))?;

        let point_cloud_frames = self.point_cloud_reader.get_frame_num();
        info!("Total frames of cloud: {}", point_cloud_frames);

        Ok((min_video_frames as usize).min(point_cloud_frames))
    }

    pub fn align_and_write(&mut self, output_path: &str) -> Result<()> {
        let align_frame_count = self
            .get_align_frame_count()
            .context("Failed to get align frame count")?;

        debug!("Align frame num: {align_frame_count}");

        for (video_idx, video_reader) in self.video_readers.iter_mut().enumerate() {
            info!("Fetching images from video {}...", video_reader.filename);

            let dir_path = Path::new(output_path).join(format!("images/images_{}", video_idx));
            fs::create_dir_all(dir_path.clone())
                .context("Failed to create directory")
                .map_err(|e| {
                    error!("Failed to create directory {:?}: {e}", dir_path);
                    e
                })?;

            let video_frame_count = video_reader.total_frames()?;
            let align_freq = video_frame_count as f64 / align_frame_count as f64;
            debug!("Video frame count: {video_frame_count}, align interval: {align_freq}");

            let pb = ProgressBar::new(align_frame_count as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );

            let mut last_frame_idx = 0usize;
            for align_idx in 1..=align_frame_count {
                let frame_idx = (align_freq * align_idx as f64).round() as usize;
                debug!(
                    "Frame index: {}, last frame index: {}",
                    frame_idx, last_frame_idx
                );

                if let Some(frame) = video_reader
                    .next_nth_frame(frame_idx - last_frame_idx)
                    .context("Failed to get next nth frame")
                    .map_err(|e| {
                        error!(
                            "Failed to get next {}th frame for video {}: {}",
                            frame_idx - last_frame_idx,
                            video_idx,
                            e
                        );
                        e
                    })?
                {
                    let image = VideoReader::convert_frame_to_image(&frame)
                        .context("Failed to convert video frame to image")
                        .map_err(|e| {
                            error!("Failed to convert video frame to image: {}", e);
                            e
                        })?;

                    image
                        .save(dir_path.join(format!("{:06}.png", align_idx - 1)))
                        .context("Failed to save image")
                        .map_err(|e| {
                            error!("Failed to save image {} for video {}", align_idx, video_idx);
                            e
                        })?;
                } else {
                    warn!(
                        "Frame {} of video {} is empty, will not be saved.",
                        frame_idx, video_idx
                    );
                }

                last_frame_idx = frame_idx;
                trace!(
                    "Successfully fetched image {} for video {}.",
                    align_idx,
                    video_idx
                );

                pb.set_position((align_idx - 1) as u64);
            }

            trace!("Successfully aligned and written video {}", video_idx);
        }

        info!(
            "Fetching point clouds from Hdf5 file {}...",
            self.point_cloud_reader.filename
        );

        let cloud_dir_path = Path::new(output_path).join("points");
        fs::create_dir_all(cloud_dir_path.clone())
            .context("Failed to create directory")
            .map_err(|e| {
                error!("Failed to create directory {:?}: {e}", cloud_dir_path);
                e
            })?;

        let align_freq = self.point_cloud_reader.get_frame_num() as f64 / align_frame_count as f64;

        let pb = ProgressBar::new(align_frame_count as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        for align_idx in 0..align_frame_count {
            let cloud_idx = (align_freq * align_idx as f64).round() as usize;
            let cloud = self
                .point_cloud_reader
                .read_pointcloud_frame(cloud_idx)
                .context("Failed to read pointcloud frame")
                .map_err(|e| {
                    error!("Failed to read pointcloud {}: {}", cloud_idx, e);
                    e
                })?;
            write_points_to_pcd(cloud, cloud_dir_path.join(format!("{:06}.pcd", align_idx)))
                .context("Failed to save cloud")
                .map_err(|e| {
                    error!("Failed to save cloud {}: {}", align_idx, e);
                    e
                })?;
            trace!("Successfully fetched cloud {}.", align_idx);

            pb.set_position(align_idx as u64);
        }
        info!("Successfully aligned and written cloud");

        Ok(())
    }
}
