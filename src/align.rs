use anyhow::{anyhow, Result};
use image::{DynamicImage, RgbImage};
use nalgebra::Point3;
use tracing::{debug, error, span, trace, warn, Level};

use crate::{
    config::SourceConfig,
    io::{hdf5::Hdf5PointCloudReader, video::VideoReader},
};

pub struct FrameAligner {
    video_readers: Vec<VideoReader>,
    video_marks: Vec<String>,
    point_cloud_reader: Hdf5PointCloudReader,
}

impl FrameAligner {
    pub fn new(
        video_file_paths: &[&str],
        video_marks: &[&str],
        pointcloud_file_path: &str,
    ) -> Result<Self> {
        let video_readers: Result<Vec<_>> = video_file_paths
            .iter()
            .map(|video_file_path| VideoReader::from_file(*video_file_path))
            .collect();

        let point_cloud_reader =
            Hdf5PointCloudReader::from_file(pointcloud_file_path).map_err(|e| {
                error!("Failed to construct point cloud reader: {e}");
                e.context("Failed to construct point cloud reader")
            })?;

        Ok(Self {
            video_readers: video_readers.map_err(|e| {
                error!("Failed to construct video readers: {e}");
                e.context("Failed to construct video readers")
            })?,
            video_marks: video_marks.iter().map(|val| val.to_string()).collect(),
            point_cloud_reader,
        })
    }

    pub fn from_config(config: &SourceConfig) -> Result<Self> {
        Self::new(
            &config
                .video
                .iter()
                .map(|config| config.file_path.as_str())
                .collect::<Vec<&str>>(),
            &config
                .video
                .iter()
                .map(|config| config.name.as_str())
                .collect::<Vec<&str>>(),
            &config.point_cloud_file_path,
        )
    }

    pub fn align_frame_count(&self) -> Result<usize> {
        let min_video_frames = self
            .video_readers
            .iter()
            .enumerate()
            .map(|(idx, reader)| {
                let frames = reader.total_frames().map_err(|e| {
                    error!("Failed to get total frames of video {}", reader.filename);
                    e.context("Failed to get total frames of video")
                })?;

                debug!("Total frames of video {}: {}", idx, frames);
                Ok(frames)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min()
            .ok_or_else(|| anyhow!("Total frames iterator is empty"))?;

        let point_cloud_frames = self.point_cloud_reader.get_frame_num();
        debug!("Total frames of cloud: {}", point_cloud_frames);

        Ok((min_video_frames as usize).min(point_cloud_frames))
    }

    #[inline]
    pub fn video_num(&self) -> usize {
        self.video_readers.len()
    }

    #[inline]
    pub fn video_marks(&self) -> Vec<String> {
        self.video_marks.clone()
    }

    pub fn aligned_frame_iter(
        &mut self,
    ) -> Result<impl Iterator<Item = (Vec<Option<DynamicImage>>, Option<Vec<Point3<f32>>>)> + '_>
    {
        let span = span!(Level::TRACE, "FrameAligner::align");
        let _enter = span.enter();

        self.video_readers
            .iter_mut()
            .enumerate()
            .try_for_each(|(idx, reader)| {
                reader.reset().map_err(|e| {
                    error!("Failed to reset video reader {idx}: {e}");
                    e.context("Failed to reset video reader")
                })
            })?;

        let align_frame_count = self.align_frame_count().map_err(|e| {
            error!("Failed to get align frame count: {e}");
            e
        })?;

        debug!("Align frame count calculated: {align_frame_count}");

        let video_align_freqs = self.calculate_video_align_freqs(align_frame_count)?;
        let pointcloud_align_freq = self.calculate_pointcloud_align_freq(align_frame_count);

        debug!(
            "Point cloud total frames: {}, align frequency: {}",
            self.point_cloud_reader.get_frame_num(),
            pointcloud_align_freq
        );

        let mut last_frame_indices: Vec<i32> = vec![-1; self.video_readers.len()];

        let iter = (0..align_frame_count).map(move |align_idx| {
            trace!("Starting alignment for frame index: {align_idx}");
            let video_frames = self.fetch_video_frames(align_idx, &video_align_freqs, &mut last_frame_indices).unwrap_or(vec![None; self.video_num()]);
            let cloud = self.fetch_pointcloud_frame(align_idx, pointcloud_align_freq).unwrap_or(None);
            trace!(
                "Finished alignment for frame index: {align_idx} (video frames: {}, point cloud: {})",
                video_frames.len(),
                if cloud.is_some() { "present" } else { "missing" }
            );
            (video_frames, cloud)
        });

        debug!("Iterator for alignment frames successfully created.");

        Ok(iter)
    }

    fn calculate_video_align_freqs(&self, align_frame_count: usize) -> Result<Vec<f64>> {
        self.video_readers
            .iter()
            .map(|reader| {
                let video_frame_count = reader.total_frames().map_err(|e| {
                    error!(
                        "Failed to get total frames from video {}: {e}",
                        reader.filename
                    );

                    e.context("Failed to get total frames from video")
                })?;

                debug!(
                    "Video '{}' total frames: {}, align frequency: {}",
                    reader.filename,
                    video_frame_count,
                    video_frame_count as f64 / align_frame_count as f64
                );

                Ok(video_frame_count as f64 / align_frame_count as f64)
            })
            .collect::<Result<Vec<_>>>()
    }

    fn calculate_pointcloud_align_freq(&self, align_frame_count: usize) -> f64 {
        self.point_cloud_reader.get_frame_num() as f64 / align_frame_count as f64
    }

    fn fetch_video_frames(
        &mut self,
        align_idx: usize,
        video_align_freqs: &[f64],
        last_frame_indices: &mut Vec<i32>,
    ) -> Result<Vec<Option<DynamicImage>>> {
        let mut video_frames = Vec::with_capacity(self.video_readers.len());

        for (video_idx, (video_reader, align_freq)) in self
            .video_readers
            .iter_mut()
            .zip(video_align_freqs.iter())
            .enumerate()
        {
            let frame_idx = (*align_freq * align_idx as f64).round() as usize;
            let frame_skip = Self::calculate_frame_skip(video_idx, frame_idx, last_frame_indices)?;

            debug!(
                "Fetching frame for video '{}', align_idx: {}, frame_idx: {}, frame_skip: {}",
                video_reader.filename, align_idx, frame_idx, frame_skip
            );

            let frame = Self::fetch_video_frame(video_reader, frame_skip, video_idx, frame_idx)?;

            video_frames.push(frame);

            last_frame_indices[video_idx] = frame_idx as i32;
        }

        Ok(video_frames)
    }

    fn calculate_frame_skip(
        video_idx: usize,
        frame_idx: usize,
        last_frame_indices: &[i32],
    ) -> Result<usize> {
        if frame_idx as i32 >= last_frame_indices[video_idx] {
            Ok((frame_idx as i32 - last_frame_indices[video_idx]) as usize)
        } else {
            error!(
                "Frame index {} is less than the last processed frame index {} for video {}.",
                frame_idx, last_frame_indices[video_idx], video_idx
            );
            Err(anyhow::anyhow!(
                "Frame index calculation error for video {}",
                video_idx
            ))
        }
    }

    fn fetch_video_frame(
        video_reader: &mut VideoReader,
        frame_skip: usize,
        video_idx: usize,
        frame_idx: usize,
    ) -> Result<Option<DynamicImage>> {
        match video_reader.next_nth_frame(frame_skip) {
            Ok(Some(frame)) => {
                debug!(
                    "Successfully fetched frame {} from video '{}'",
                    frame_idx, video_reader.filename
                );
                Ok(
                    RgbImage::from_raw(frame.width(), frame.height(), frame.data(0).to_vec())
                        .map_or(None, |image| Some(DynamicImage::ImageRgb8(image))),
                )
            }
            Ok(None) => {
                warn!("Frame {} of video {} is empty.", frame_idx, video_idx);
                Ok(None)
            }
            Err(e) => {
                error!(
                    "Failed to get frame {} for video {}: {}",
                    frame_idx, video_idx, e
                );
                Err(e)
            }
        }
    }

    fn fetch_pointcloud_frame(
        &self,
        align_idx: usize,
        pointcloud_align_freq: f64,
    ) -> Result<Option<Vec<Point3<f32>>>> {
        let cloud_idx = (pointcloud_align_freq * align_idx as f64).round() as usize;
        debug!(
            "Fetching point cloud for align_idx: {}, cloud_idx: {}",
            align_idx, cloud_idx
        );

        match self.point_cloud_reader.read_pointcloud_frame(cloud_idx) {
            Ok(cloud) => {
                debug!("Successfully fetched point cloud frame {}", cloud_idx);
                Ok(Some(cloud))
            }
            Err(e) => {
                error!("Failed to read pointcloud frame {}: {}", cloud_idx, e);
                Err(e)
            }
        }
    }
}
