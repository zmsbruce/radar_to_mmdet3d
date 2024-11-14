use std::{
    fs::{self, File},
    path::PathBuf,
};

use chrono::Local;
use mmdet3dgen::radar::Radar;

use anyhow::{Context, Result};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

fn main() -> Result<()> {
    init_logging("log")?;

    let radar =
        Radar::new("assets/car.onnx", "assets/armor.onnx").context("Failed to initialize radar")?;

    let image =
        image::open(PathBuf::from("assets/test/frame.jpg")).context("Failed to read image")?;
    let point_cloud = vec![];

    let result = radar
        .run_rdlt(&image, &point_cloud)
        .context("Failed to run rdlt")?;

    println!("Result: {:#?}", result);

    Ok(())
}

fn init_logging(log_dir: &str) -> Result<()> {
    if !fs::exists(log_dir)? {
        fs::create_dir_all(log_dir).context("Failed to create directory")?;
    }

    let file_name = format!(
        "{}/mmdet3dgen_{}.log",
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
