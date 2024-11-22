use anyhow::{anyhow, Context, Result};
use gstreamer::{prelude::*, Pipeline, Sample};
use gstreamer_app::AppSink;
use gstreamer_video::{VideoFrameExt, VideoFrameRef, VideoInfo};
use image::RgbImage;

pub struct VideoReader {
    appsink: AppSink,
    counter: usize,
}

impl VideoReader {
    pub fn new(video_path: &str) -> Result<Self> {
        gstreamer::init().context("Failed to initialize gstreamer")?;

        let pipeline = gstreamer::parse::launch(&format!(
            "filesrc location={} ! decodebin ! videoconvert ! appsink name=appsink",
            video_path
        ))
        .context("Failed to parse launch")?
        .downcast::<Pipeline>()
        .unwrap();

        let appsink = pipeline
            .by_name("appsink")
            .ok_or_else(|| anyhow!("Failed to get appsink from pipeline"))?
            .downcast::<AppSink>()
            .unwrap();

        appsink.set_caps(Some(
            &gstreamer::Caps::builder("video/x-raw")
                .field("format", &"RGB")
                .build(),
        ));
        appsink.set_property("sync", &false);

        pipeline.set_state(gstreamer::State::Playing)?;

        Ok(Self {
            appsink,
            counter: 0,
        })
    }
}

impl VideoReader {
    pub fn get_frame_from_sample(sample: &Sample) -> Result<RgbImage> {
        let buffer = sample
            .buffer()
            .ok_or_else(|| anyhow!("Failed to get buffer"))?;
        let caps = sample.caps().ok_or_else(|| anyhow!("Failed to get caps"))?;
        let info = VideoInfo::from_caps(&caps).context("Failed to get VideoInfo from caps")?;

        let frame = VideoFrameRef::from_buffer_ref_readable(buffer, &info)
            .context("Failed to get VideoFrameRef from buffer")?;

        let (width, height) = (frame.width(), frame.height());
        let data = frame.plane_data(0)?;

        let image = RgbImage::from_raw(width, height, data.to_vec())
            .ok_or_else(|| anyhow!("Failed to create image from frame"))?;

        Ok(image)
    }
}

impl Iterator for VideoReader {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = self.appsink.pull_sample().ok();
        if sample.is_some() {
            self.counter += 1;
        }

        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_reader() -> Result<()> {
        let video_path = "assets/test/test.mp4";
        let mut reader = VideoReader::new(video_path)?;

        let sample = reader
            .next()
            .ok_or_else(|| anyhow!("Failed to get sample"))?;
        assert_eq!(reader.counter, 1);

        let frame = VideoReader::get_frame_from_sample(&sample)?;
        assert_eq!(frame.width(), 640);
        assert_eq!(frame.height(), 640);

        assert!(frame.pixels().all(|pixel| pixel.0 == [255, 21, 0]));

        Ok(())
    }
}
