use std::fs;

use anyhow::{Context, Ok, Result};
use serde::Deserialize;
use tracing::{debug, span, trace, Level};

use super::detect::config::DetectorConfig;

#[derive(Debug, Deserialize)]
pub struct RadarConfig {
    pub detect: DetectorConfig,
}

impl RadarConfig {
    pub fn from_file(filename: &str) -> Result<Self> {
        let span = span!(Level::TRACE, "RadarConfig::from_file");
        let _enter = span.enter();

        trace!("Reading content from file {filename}...");
        let config_content =
            fs::read_to_string(filename).context("Failed to read config from file")?;

        trace!("Deserializing content to RadarConfig...");
        let config: Self = toml::from_str(&config_content)
            .context("Failed to deserialize content to RadarConfig")?;

        debug!("Configurations: {:#?}", config);
        Ok(config)
    }
}
