use std::fs::{self, File};

use anyhow::{Context, Result};
use chrono::Local;
use tracing::{span, Level};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

fn main() -> Result<()> {
    init_logging("log")?;

    let span = span!(Level::TRACE, "main");
    let _enter = span.enter();

    println!("Hello, world!");

    Ok(())
}

fn init_logging(log_dir: &str) -> Result<()> {
    if !fs::exists(log_dir)? {
        fs::create_dir_all(log_dir).context("Failed to create directory")?;
    }

    let file_name = format!(
        "{}/rm_radar_to_mmdet3d_{}.log",
        log_dir,
        Local::now().format("%Y-%m-%d_%H-%M-%S")
    );

    let file = File::create(&file_name).context("Failed to create file")?;
    let file_appender = fmt::layer().with_writer(file).with_ansi(false);
    let file_filter = EnvFilter::new("trace");

    let console_appender = fmt::layer().with_writer(std::io::stdout).with_ansi(true);
    let console_filter = EnvFilter::new("info");

    tracing_subscriber::registry()
        .with(console_appender.with_filter(console_filter))
        .with(file_appender.with_filter(file_filter))
        .try_init()
        .context("Failed to initialize tracing subscriber")?;

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
