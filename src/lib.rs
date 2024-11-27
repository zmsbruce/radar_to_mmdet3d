use std::{path::PathBuf, sync::Arc, time::Duration};

use align::FrameAligner;
use anyhow::Result;
use futures::{future, StreamExt};
use image::{DynamicImage, GenericImageView};
use indicatif::{ProgressBar, ProgressStyle};
use io::{pcd::save_pointcloud, save_image};
use radar::{
    detect::{RobotDetection, RobotDetector},
    locate::Locator,
};
use rayon::prelude::*;
use tokio::{fs, task};
use tracing::{error, warn};

pub mod align;
pub mod config;
pub mod io;
pub mod radar;

pub async fn create_output_dirs(root_dir: &str, image_num: usize) -> Result<()> {
    let root_dir = PathBuf::from(root_dir);
    let image_dirs: Vec<_> = (0..image_num)
        .map(|val| root_dir.join(format!("images/images_{val}")))
        .collect();
    let pointcloud_dir = root_dir.join("points");

    for dir in image_dirs {
        fs::create_dir_all(&dir).await.map_err(|e| {
            error!("Failed to create directory {:?}: {e}", dir);
            e
        })?;
    }
    fs::create_dir_all(&pointcloud_dir).await.map_err(|e| {
        error!("Failed to create directory {:?}: {e}", pointcloud_dir);
        e
    })?;

    Ok(())
}

pub async fn detect_image(
    detector: Arc<RobotDetector>,
    image: Arc<DynamicImage>,
) -> Result<Vec<RobotDetection>> {
    let image = image.clone();
    task::spawn_blocking(move || {
        detector.detect(&image).map_err(|e| {
            error!("Failed to detect image: {e}");
            e
        })
    })
    .await?
}

pub fn build_models(detectors: &mut Vec<RobotDetector>) -> Result<()> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .map_err(|e| {
                error!("Failed to set spinner template: {e}");
                e
            })?,
    );
    spinner.set_message("Building models of robot detector...");
    spinner.enable_steady_tick(Duration::from_millis(100));

    detectors
        .par_iter_mut()
        .map(|detector| detector.build_models())
        .collect::<Result<(), _>>()
        .map_err(|e| {
            error!("Failed to build models: {e}");
            spinner.finish_with_message("Failed to build models.");
            e
        })?;

    spinner.finish_and_clear();
    Ok(())
}

pub async fn save_aligned_frames_with_detect_and_background_update(
    aligner: &mut FrameAligner,
    detectors: &Vec<Arc<RobotDetector>>,
    locators: &mut Vec<Locator>,
    root_dir: &str,
) -> Result<Vec<Vec<Option<Vec<RobotDetection>>>>> {
    let align_frame_count = aligner.align_frame_count().await.map_err(|e| {
        error!("Failed to get align frame count: {e}");
        e
    })?;
    let progress_bar = ProgressBar::new(align_frame_count as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let align_stream = aligner.aligned_frame_stream().await.map_err(|e| {
        error!("Failed to get aligned frame stream: {e}");
        e
    })?;

    let root_dir = root_dir.to_string();

    let results = align_stream
        .enumerate()
        .then(|(frame_idx, (images, point_cloud))| {
            progress_bar.set_position(frame_idx as u64);

            if let Some(point_cloud) = point_cloud {
                locators.par_iter_mut().enumerate().for_each(|(idx, locator)| {
                    let image_size = images.get(idx).and_then(|image| image.as_ref().map(|i| i.dimensions()));
                    if let Some(image_size) = image_size {
                        if let Err(e) = locator.update_background_depth_map(&point_cloud, image_size) {
                            error!("Failed to update background depth map: {e}");
                        }
                    } else {
                        warn!("Image {idx} of frame {frame_idx} is empty, will skip background depth map update.");
                    }
                });

                task::spawn({
                    let point_cloud = point_cloud.clone();
                    let root_dir = root_dir.clone();
                    async move {
                        if let Err(e) = save_pointcloud(
                            point_cloud,
                            PathBuf::from(&root_dir).join(format!("points/{:06}.pcd", frame_idx)),
                        ).await {
                            error!("Failed to save point cloud of frame {frame_idx}: {e}");
                        }
                    }
                });
            }

            let root_dir = root_dir.clone();
            async move {
                let detection_futures = images.iter().enumerate().map(|(image_idx, image)| {
                    let detector = detectors[image_idx].clone();
                    let image = image.clone();
                    let root_dir = root_dir.clone();

                    async move {
                        if let Some(image) = image {
                            task::spawn({
                                let image = image.clone();
                                async move {
                                    if let Err(e) = save_image(
                                        &image,
                                        PathBuf::from(&root_dir).join(format!("images/images_{image_idx}/{:06}.png", frame_idx))
                                    ).await {
                                        error!("Failed to save image {image_idx} of frame {frame_idx}: {e}");
                                    }
                                }
                            });
                    
                            detector.detect(&image).map_err(|e| {
                                error!("Failed to detect image: {e}");
                                e
                            }).ok()
                        } else {
                            warn!("Image {image_idx} of frame {frame_idx} is empty, will skip detection and save.");
                            None
                        }
                    }
                    
                });

                future::join_all(detection_futures).await
            }
        })
        .collect()
        .await;

    progress_bar.finish_and_clear();
    Ok(results)
}
