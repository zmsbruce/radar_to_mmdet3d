use anyhow::{anyhow, Result};
use image::RgbImage;
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
    pub fn new(video_file_paths: &[&str], video_marks: &[&str], pointcloud_file_path: &str) -> Result<Self> {
        let video_readers: Result<Vec<_>> = video_file_paths
            .iter()
            .map(|video_file_path| VideoReader::from_file(*video_file_path))
            .collect();

        let point_cloud_reader =
            Hdf5PointCloudReader::from_file(pointcloud_file_path).map_err(|e| {
                error!("Failed to construct point cloud reader: {e}");
                e
            })?;

        Ok(Self {
            video_readers: video_readers.map_err(|e| {
                error!("Failed to construct video readers: {e}");
                e
            })?,
            video_marks: video_marks.iter().map(|val| val.to_string()).collect(),
            point_cloud_reader,
        })
    }

    pub fn from_config_file<P>(file_path: P) -> Result<Self>
    where
        P: AsRef<std::path::Path> + std::fmt::Debug,
    {
        let config = SourceConfig::from_file(file_path).map_err(|e| {
            error!("Failed to read aligner config file: {}", e);
            e
        })?;

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
                    e
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

    pub fn align(
        &mut self,
    ) -> Result<impl Iterator<Item = Result<(Vec<Option<RgbImage>>, Option<Vec<Point3<f32>>>)>> + '_>
    {
        let span = span!(Level::TRACE, "FrameAligner::align");
        let _enter = span.enter();

        let align_frame_count = self.align_frame_count().map_err(|e| {
            error!("Failed to get align frame count: {e}");
            e
        })?;

        debug!("Align frame count calculated: {align_frame_count}");

        let video_align_freqs = self
            .video_readers
            .iter()
            .map(|reader| {
                let video_frame_count = reader.total_frames().map_err(|e| {
                    error!(
                        "Failed to get total frames from video {}: {e}",
                        reader.filename
                    );
                    e
                })?;
                debug!(
                    "Video '{}' total frames: {}, align frequency: {}",
                    reader.filename,
                    video_frame_count,
                    video_frame_count as f64 / align_frame_count as f64
                );
                Ok(video_frame_count as f64 / align_frame_count as f64)
            })
            .collect::<Result<Vec<_>>>()?;

        let pointcloud_align_freq =
            self.point_cloud_reader.get_frame_num() as f64 / align_frame_count as f64;

        debug!(
            "Point cloud total frames: {}, align frequency: {}",
            self.point_cloud_reader.get_frame_num(),
            pointcloud_align_freq
        );

        let mut last_frame_indices: Vec<i32> = vec![-1; self.video_readers.len()];

        let iter = (0..align_frame_count).map(move |align_idx| {
            trace!("Starting alignment for frame index: {align_idx}");

            let mut video_frames = Vec::with_capacity(self.video_readers.len());

            for (video_idx, (video_reader, align_freq)) in self
                .video_readers
                .iter_mut()
                .zip(video_align_freqs.iter())
                .enumerate()
            {
                let frame_idx = (*align_freq * align_idx as f64).round() as usize;
                let frame_skip = if frame_idx as i32 >= last_frame_indices[video_idx] {
                    (frame_idx as i32 - last_frame_indices[video_idx]) as usize
                } else {
                    error!(
                        "Frame index {} is less than the last processed frame index {} for video '{}'.",
                        frame_idx, last_frame_indices[video_idx], video_reader.filename
                    );
                    return Err(anyhow::anyhow!(
                        "Frame index calculation error for video '{}'",
                        video_reader.filename
                    ));
                };
    
                debug!(
                    "Fetching frame for video '{}', align_idx: {}, frame_idx: {}, frame_skip: {}",
                    video_reader.filename, align_idx, frame_idx, frame_skip
                );

                let frame = match video_reader.next_nth_frame(frame_skip) {
                    Ok(Some(frame)) => {
                        debug!(
                            "Successfully fetched frame {} from video '{}'",
                            frame_idx, video_reader.filename
                        );
                        RgbImage::from_raw(frame.width(), frame.height(), frame.data(0).to_vec())
                    }
                    Ok(None) => {
                        warn!("Frame {} of video {} is empty.", frame_idx, video_idx);
                        None
                    }
                    Err(e) => {
                        error!(
                            "Failed to get frame {} for video {}: {}",
                            frame_idx, video_idx, e
                        );
                        return Err(e);
                    }
                };

                trace!(
                    "Processed frame {} for video '{}' (align_idx: {})",
                    frame_idx,
                    video_reader.filename,
                    align_idx
                );

                video_frames.push(frame);

                last_frame_indices[video_idx] = frame_idx as i32;
            }

            let cloud_idx = (pointcloud_align_freq * align_idx as f64).round() as usize;
            debug!(
                "Fetching point cloud for align_idx: {}, cloud_idx: {}",
                align_idx, cloud_idx
            );

            let cloud = match self.point_cloud_reader.read_pointcloud_frame(cloud_idx) {
                Ok(cloud) => {
                    debug!("Successfully fetched point cloud frame {}", cloud_idx);
                    Some(cloud)
                }
                Err(e) => {
                    error!("Failed to read pointcloud frame {}: {}", cloud_idx, e);
                    return Err(e);
                }
            };

            trace!(
                "Finished alignment for frame index: {align_idx} (video frames: {}, point cloud: {})",
                video_frames.len(),
                if cloud.is_some() { "present" } else { "missing" }
            );

            Ok((video_frames, cloud))
        });

        debug!("Iterator for alignment frames successfully created.");

        Ok(iter)
    }
}
