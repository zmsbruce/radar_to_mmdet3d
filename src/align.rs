use std::sync::Arc;

use anyhow::{anyhow, Result};
use futures::{stream, Stream, StreamExt};
use image::{DynamicImage, RgbImage};
use nalgebra::Point3;
use tokio::{sync::Mutex, task};
use tracing::{debug, error, span, trace, warn, Level};

use crate::{
    config::SourceConfig,
    io::{hdf5::Hdf5PointCloudReader, video::VideoReader},
};

pub struct FrameAligner {
    video_readers: Arc<Mutex<Vec<VideoReader>>>,
    video_marks: Vec<String>,
    point_cloud_reader: Arc<Hdf5PointCloudReader>,
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
                e
            })?;

        Ok(Self {
            video_readers: Arc::new(Mutex::new(video_readers.map_err(|e| {
                error!("Failed to construct video readers: {e}");
                e
            })?)),
            video_marks: video_marks.iter().map(|val| val.to_string()).collect(),
            point_cloud_reader: Arc::new(point_cloud_reader),
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

    pub async fn align_frame_count(&self) -> Result<usize> {
        let min_video_frames = self
            .video_readers
            .lock()
            .await
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
    pub async fn video_num(&self) -> usize {
        self.video_readers.lock().await.len()
    }

    #[inline]
    pub fn video_marks(&self) -> Vec<String> {
        self.video_marks.clone()
    }

    pub async fn aligned_frame_stream(
        &mut self,
    ) -> Result<
        impl Stream<
                Item = (
                    Vec<Option<Arc<DynamicImage>>>,
                    Option<Arc<Vec<Point3<f32>>>>,
                ),
            > + '_,
    > {
        let span = span!(Level::TRACE, "FrameAligner::align");
        let _enter = span.enter();

        let align_frame_count = self.align_frame_count().await.map_err(|e| {
            error!("Failed to get align frame count: {e}");
            e
        })?;
        debug!("Align frame count calculated: {align_frame_count}");

        let video_readers = self.video_readers.lock().await;
        let video_align_freqs =
            Arc::new(self.calculate_video_align_freqs(&video_readers, align_frame_count)?);
        let video_readers_len = video_readers.len();
        drop(video_readers);

        let pointcloud_align_freq =
            self.point_cloud_reader.get_frame_num() as f64 / align_frame_count as f64;
        debug!(
            "Point cloud total frames: {}, align frequency: {}",
            self.point_cloud_reader.get_frame_num(),
            pointcloud_align_freq
        );

        let last_frame_indices = Arc::new(Mutex::new(vec![-1; video_readers_len]));

        let iter = stream::iter(0..align_frame_count).then(move |align_idx| {
            Self::process_frame(
                align_idx,
                video_align_freqs.clone(),
                pointcloud_align_freq,
                self.video_readers.clone(),
                self.point_cloud_reader.clone(),
                last_frame_indices.clone(),
            )
        });

        debug!("Iterator for alignment frames successfully created.");
        Ok(iter)
    }

    fn calculate_video_align_freqs(
        &self,
        video_readers: &[VideoReader],
        align_frame_count: usize,
    ) -> Result<Vec<f64>> {
        video_readers
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
            .collect::<Result<Vec<_>>>()
    }

    async fn process_frame(
        align_idx: usize,
        video_align_freqs: Arc<Vec<f64>>,
        pointcloud_align_freq: f64,
        video_readers: Arc<Mutex<Vec<VideoReader>>>,
        point_cloud_reader: Arc<Hdf5PointCloudReader>,
        last_frame_indices: Arc<Mutex<Vec<i32>>>,
    ) -> (
        Vec<Option<Arc<DynamicImage>>>,
        Option<Arc<Vec<Point3<f32>>>>,
    ) {
        trace!("Starting alignment for frame index: {align_idx}");

        let cloud_idx = (pointcloud_align_freq * align_idx as f64).round() as usize;
        debug!(
            "Fetching point cloud for align_idx: {}, cloud_idx: {}",
            align_idx, cloud_idx
        );

        let (video_frames, point_cloud) = tokio::join!(
            async {
                let frames = Self::fetch_video_frames(
                    align_idx,
                    video_align_freqs,
                    video_readers.clone(),
                    last_frame_indices.clone(),
                )
                .await
                .map_err(|e| {
                    error!("Failed to fetch video frames: {e}");
                    e
                })
                .unwrap_or(vec![None; 3]);
                frames
            },
            async {
                let result = task::spawn_blocking({
                    let point_cloud_reader = Arc::clone(&point_cloud_reader);
                    move || point_cloud_reader.read_pointcloud_frame(cloud_idx)
                })
                .await;

                match result {
                    Ok(Ok(cloud)) => Some(Arc::new(cloud)),
                    Ok(Err(e)) => {
                        error!("Failed to read pointcloud frame {}: {}", cloud_idx, e);
                        None
                    }
                    Err(e) => {
                        error!("Failed to join future: {e}");
                        None
                    }
                }
            }
        );

        trace!("Finished alignment for frame index: {align_idx}");
        (video_frames, point_cloud)
    }

    async fn fetch_video_frames(
        align_idx: usize,
        video_align_freqs: Arc<Vec<f64>>,
        video_readers: Arc<Mutex<Vec<VideoReader>>>,
        last_frame_indices: Arc<Mutex<Vec<i32>>>,
    ) -> Result<Vec<Option<Arc<DynamicImage>>>> {
        let mut video_readers = video_readers.lock().await;
        let mut last_frame_indices = last_frame_indices.lock().await;

        let mut video_frames = Vec::with_capacity(video_readers.len());
        for (video_idx, (video_reader, align_freq)) in video_readers
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

            let frame = Self::fetch_single_frame(video_reader, frame_skip, frame_idx, video_idx)?;
            video_frames.push(frame.map_or(None, |frame| Some(Arc::new(frame))));

            last_frame_indices[video_idx] = frame_idx as i32;
        }

        Ok(video_frames)
    }

    fn fetch_single_frame(
        video_reader: &mut VideoReader,
        frame_skip: usize,
        frame_idx: usize,
        video_idx: usize,
    ) -> Result<Option<DynamicImage>> {
        match video_reader.next_nth_frame(frame_skip) {
            Ok(Some(frame)) => {
                debug!(
                    "Successfully fetched frame {} from video '{}'",
                    frame_idx, video_reader.filename
                );
                let image =
                    RgbImage::from_raw(frame.width(), frame.height(), frame.data(0).to_vec())
                        .ok_or_else(|| anyhow!("Container is not big enough"))?;

                Ok(Some(DynamicImage::ImageRgb8(image)))
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
}
