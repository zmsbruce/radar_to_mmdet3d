use std::{io::Cursor, path::PathBuf};

use anyhow::Result;
use image::DynamicImage;
use tokio::{fs::File, io::AsyncWriteExt};
use tracing::error;

pub mod hdf5;
pub mod pcd;
pub mod video;

pub async fn save_image(image: DynamicImage, path: PathBuf) -> Result<()> {
    let mut buffer = Cursor::new(Vec::new());
    let format = image::ImageFormat::Png;

    image.write_to(&mut buffer, format).map_err(|e| {
        error!("Failed to write image to buffer: {e}");
        e
    })?;

    let mut file = File::create(path).await.map_err(|e| {
        error!("Failed to create image file: {e}");
        e
    })?;
    file.write_all(&buffer.into_inner()).await.map_err(|e| {
        error!("Failed to write image buffer to file");
        e
    })?;

    Ok(())
}
