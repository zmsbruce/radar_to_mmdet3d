use std::{
    fs::{self, File},
    io::{BufWriter, Write as _},
    path::PathBuf,
};

use align::FrameAligner;
use anyhow::Result;
use image::GenericImageView;
use indicatif::{ProgressBar, ProgressStyle};
use io::pcd::save_pointcloud;
use radar::{
    detect::{RobotDetection, RobotDetector},
    locate::Locator,
};
use rayon::prelude::*;
use tracing::{error, warn};

pub mod align;
pub mod config;
pub mod io;
pub mod radar;

pub fn create_output_dirs(root_dir: &str, image_num: usize) -> Result<()> {
    let root_dir = PathBuf::from(root_dir);
    let image_dirs: Vec<_> = (0..image_num)
        .map(|val| root_dir.join(format!("images/images_{val}")))
        .collect();
    let pointcloud_dir = root_dir.join("points");
    let label_dir = root_dir.join("labels");

    for dir in image_dirs {
        fs::create_dir_all(&dir).map_err(|e| {
            error!("Failed to create directory {:?}: {e}", dir);
            e
        })?;
    }
    fs::create_dir_all(&pointcloud_dir).map_err(|e| {
        error!("Failed to create directory {:?}: {e}", pointcloud_dir);
        e
    })?;
    fs::create_dir_all(&label_dir).map_err(|e| {
        error!("Failed to create directory {:?}: {e}", label_dir);
        e
    })?;

    Ok(())
}

pub fn build_model(detector: &mut RobotDetector) -> Result<()> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .map_err(|e| {
                error!("Failed to set spinner template: {e}");
                e
            })?,
    );
    spinner.set_message("Building models for robot detector...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    detector
        .build_models()
        .map_err(|e| {
            error!("Failed to build models: {e}");
            spinner.finish_with_message("Failed to build models.");
            e
        })?;

    spinner.finish_with_message("Finished building models.");
    Ok(())
}

pub fn process_and_save_aligned_frames(
    aligner: &mut FrameAligner,
    detector: &RobotDetector,
    locators: &mut Vec<Locator>,
    root_dir: &str,
) -> Result<Vec<Vec<Option<Vec<RobotDetection>>>>> {
    let align_frame_count = aligner.align_frame_count().map_err(|e| {
        error!("Failed to get align frame count: {e}");
        e
    })?;

    let progress_bar = ProgressBar::new(align_frame_count as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    progress_bar.set_message("Processing and saving frames...");

    let root_dir = PathBuf::from(root_dir);

    let detect_results = aligner
        .aligned_frame_iter()
        .map_err(|e| {
            error!("Failed to extract iterator for aligner: {e}");
            e
        })?
        .enumerate()
        .map(|(frame_idx, (images, point_cloud))| {
            progress_bar.set_position(frame_idx as u64);

            if let Some(point_cloud) = point_cloud {
                let point_cloud: Vec<_> = point_cloud.into_par_iter().map(|point| point * 1000.0).collect();
                locators.par_iter_mut().enumerate().for_each(|(idx, locator)| {
                    if let Some(image_size) = images[idx]
                        .as_ref()
                        .and_then(|image| Some(image.dimensions()))
                    {
                        if let Err(e) = locator.update_background_depth_map(&point_cloud, image_size) {
                            error!("Failed to update background depth map for frame {frame_idx}: {e}");
                        }
                    } else {
                        warn!("Image {idx} of frame {frame_idx} is empty, skipped background depth map update.");
                    }
                });

                if let Err(e) = save_pointcloud(
                    &point_cloud,
                    root_dir.join(format!("points/{:06}.pcd", frame_idx)),
                ) {
                    error!("Failed to save point cloud of frame {frame_idx}: {e}");
                }
            } else {
                warn!("Point cloud of frame {frame_idx} is empty, skipped background depth map update.");
                warn!("Point cloud of frame {frame_idx} is empty, skipped point cloud save.");
            }

            let detections = images.iter().enumerate().map(|(idx, image)| {
                if let Some(image) = image {
                    detector.detect(image).map_err(|e| {
                        error!("Failed to detect image {idx} of frame {frame_idx}: {e}");
                        e
                    }).ok()
                } else {
                    warn!("Image {idx} of frame {frame_idx} is empty, skipped detect.");
                    None
                }
            }).collect::<Vec<_>>();

            images.into_iter().enumerate().for_each(|(idx, image)| {
                if let Some(image) = image {
                    if let Err(e) = image.save(root_dir.join(format!("images/images_{idx}/{:06}.png", frame_idx))) {
                        error!("Failed to save image {idx} of frame {frame_idx}: {e}");
                    }
                } else {
                    warn!("Image {idx} of frame {frame_idx} is empty, skipped image save.");
                }
            });

            detections
        })
        .collect::<Vec<_>>();

    progress_bar.finish_with_message("Finished frame processing and saving.");
    Ok(detect_results)
}

pub fn locate_and_save_results(
    detect_results_frames: Vec<Vec<Option<Vec<RobotDetection>>>>,
    aligner: &mut FrameAligner,
    locators: &mut Vec<Locator>,
    root_dir: &str,
) -> Result<()> {
    let progress_bar = ProgressBar::new(detect_results_frames.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    progress_bar.set_message("Locating and saving results...");

    let root_dir = PathBuf::from(root_dir);

    let aligner_iter = aligner.aligned_frame_iter().map_err(|e| {
        error!("Failed to extract iterator for aligner: {e}");
        e
    })?;

    detect_results_frames
        .into_iter()
        .zip(aligner_iter)
        .enumerate()
        .map(|(frame_idx, (detect_results, (_, point_cloud)))| {
            assert_eq!(detect_results.len(), locators.len());

            progress_bar.set_position(frame_idx as u64);

            let locate_results = if let Some(point_cloud) = point_cloud {
                let point_cloud: Vec<_> = point_cloud
                    .into_par_iter()
                    .map(|point| point * 1000.0)
                    .collect();
            
                let locate_results = detect_results
                    .iter()
                    .zip(locators.iter_mut())
                    .enumerate()
                    .map(|(idx, (detect_result, locator))| {
                        detect_result.as_ref().map_or_else(
                            || {
                                warn!(
                                    "Detect result {idx} of frame {frame_idx} is none, skipped locate"
                                );
                                None
                            },
                            |detect_result| {
                                locator
                                    .locate_detections(&point_cloud, &detect_result)
                                    .map_err(|e| {
                                        error!(
                                            "Failed to locate detection {idx} of frame {frame_idx}: {e}"
                                        );
                                        e
                                    })
                                    .ok()
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                
                Some(locate_results)
            } else {
                None
            };

            (frame_idx, (locate_results, detect_results))
        })
        .for_each(|(frame_idx, (locate_results, detect_results))| {
            let file_path = root_dir.join(format!("labels/{:06}.txt", frame_idx));
            let file = match File::create(&file_path) {
                Ok(file) => file,
                Err(e) => {
                    error!("Failed to create {:?}: {e}", file_path);
                    return;
                }
            };
            
            if let Some(locate_results) = locate_results {
                let mut writer = BufWriter::new(file);
                locate_results
                    .into_iter()
                    .zip(detect_results.into_iter())
                    .for_each(|(locate_result, detect_result)| {
                        if locate_result.is_some() && detect_result.is_some() {
                            locate_result
                                .unwrap()
                                .into_iter()
                                .zip(detect_result.unwrap().into_iter())
                                .for_each(|(single_locate_result, single_detct_result)| {
                                    if let Some(single_locate_result) = single_locate_result {
                                        let line = format!(
                                            "{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {}\n",
                                            single_locate_result.center.x,
                                            single_locate_result.center.y,
                                            single_locate_result.center.z,
                                            single_locate_result.depth,
                                            single_locate_result.width,
                                            single_locate_result.height,
                                            0.0,
                                            single_detct_result.label.name_abbr()
                                        );
                                        if let Err(e) = writer.write_all(line.as_bytes()) {
                                            error!("Failed to write to buffer: {e}");
                                        }
                                    }
                                });
                        }
                    });
            }
            
        });

    progress_bar.finish_with_message("Finished locating and saving results.");
    Ok(())
}
