use std::fs::{self, File};

use anyhow::{anyhow, Result};
use chrono::Local;
use radar_to_mmdet3d::{
    align::FrameAligner,
    build_model,
    config::{RadarConfig, SourceConfig},
    create_output_dirs, locate_and_save_results, process_and_save_aligned_frames,
    radar::{detect::RobotDetector, locate::Locator},
};
use tracing::{error, span, Level};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

fn main() -> Result<()> {
    init_logging("log")?;

    let span = span!(Level::TRACE, "main");
    let _enter = span.enter();

    let source_config = SourceConfig::from_file("config/source.toml").map_err(|e| {
        error!("Failed to read source config: {e}");
        e
    })?;
    let mut aligner = FrameAligner::from_config(&source_config).map_err(|e| {
        error!("Failed to initialize frame aligner from config file: {e}");
        e
    })?;

    let num_videos = aligner.video_num();
    create_output_dirs(&source_config.output_dir_path, num_videos).map_err(|e| {
        error!("Failed to create output directories: {e}");
        e
    })?;

    let radar_config = RadarConfig::from_file("config/radar.toml").map_err(|e| {
        error!("Failed to load radar configuration: {e}");
        e
    })?;
    let mut detector = RobotDetector::from_config(&radar_config.detect).map_err(|e| {
        error!("Failed to initialize detector from config: {e}");
        e
    })?;
    build_model(&mut detector).map_err(|e| {
        error!("Failed to build detector model: {e}");
        e
    })?;

    let mut locators = aligner
        .video_marks()
        .into_iter()
        .map(|mark| {
            let instance_config = radar_config
                .instances
                .iter()
                .find(|instance_config| instance_config.name == mark)
                .ok_or_else(|| anyhow!("Failed to find instance config for mark {mark}"))?;

            Locator::from_config(&radar_config.locate, instance_config).map_err(|e| {
                error!("Failed to create locator for mark {mark}: {e}");
                e
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let detect_result = process_and_save_aligned_frames(
        &mut aligner,
        &detector,
        &mut locators,
        &source_config.output_dir_path,
    )
    .map_err(|e| {
        error!("Failed process and save frames: {e}");
        e
    })?;

    locate_and_save_results(
        detect_result,
        &mut aligner,
        &mut locators,
        &source_config.output_dir_path,
    )
    .map_err(|e| {
        error!("Failed to locate and save results: {e}");
        e
    })?;

    Ok(())
}

fn init_logging(log_dir: &str) -> Result<()> {
    if !fs::exists(log_dir)? {
        fs::create_dir_all(log_dir).map_err(|e| {
            error!("Failed to create directories {log_dir}: {e}");
            e
        })?;
    }

    let file_name = format!(
        "{}/radar_to_mmdet3d_{}.log",
        log_dir,
        Local::now().format("%Y-%m-%d_%H-%M-%S")
    );

    let file = File::create(&file_name).map_err(|e| {
        error!("Failed to create file {file_name}: {e}");
        e
    })?;
    let file_appender = fmt::layer().with_writer(file).with_ansi(false);
    let file_filter = EnvFilter::new("info,radar_to_mmdet3d=trace");

    let console_appender = fmt::layer().with_writer(std::io::stdout).with_ansi(true);
    let console_filter = EnvFilter::new("warn,radar_to_mmdet3d=info");

    tracing_subscriber::registry()
        .with(console_appender.with_filter(console_filter))
        .with(file_appender.with_filter(file_filter))
        .try_init()
        .map_err(|e| {
            error!("Failed to initialize tracing subscriber: {e}");
            e
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io::Read};
    use tempfile::tempdir;
    use tracing::trace;

    #[test]
    fn test_init_logging_creates_log_file() -> Result<()> {
        let tmp_dir = tempdir()?;
        let log_dir = tmp_dir.path().to_str().unwrap();

        init_logging(log_dir)?;

        let entries: Vec<_> = fs::read_dir(log_dir)?
            .map(|res| res.map(|e| e.path()))
            .collect::<Result<Vec<_>, std::io::Error>>()?;

        assert_eq!(entries.len(), 1);
        assert!(entries[0].to_str().unwrap().ends_with(".log"));

        trace!("This is a trace log.");

        let mut log_file = File::open(&entries[0])?;
        let mut contents = String::new();
        log_file.read_to_string(&mut contents)?;

        assert!(!contents.is_empty() && contents.contains("This is a trace log."));

        Ok(())
    }
}
