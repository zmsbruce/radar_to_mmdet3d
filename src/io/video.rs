use anyhow::{anyhow, Result};
use ffmpeg_next::{
    self as ffmpeg, decoder, error::EAGAIN, format::context, frame, software::scaling,
};
use std::sync::Once;
use tracing::{debug, error, span, trace, warn, Level};

static FFMPEG_INIT: Once = Once::new();

pub struct VideoReader {
    context: context::Input,
    decoder: decoder::Video,
    stream_index: usize,
    scaler: scaling::Context,
    pub filename: String,
}

impl VideoReader {
    pub fn from_file<P: AsRef<std::path::Path>>(file_path: P) -> Result<Self> {
        let span = span!(Level::TRACE, "VideoReader::from_file");
        let _enter = span.enter();

        debug!(
            "Initializing VideoReader for file: {:?}",
            file_path.as_ref()
        );

        trace!("Initializing ffmpeg");
        Self::initialize_ffmpeg();

        trace!("Opening video file to input context");
        let context = ffmpeg::format::input(&file_path).map_err(|e| {
            error!("Failed to open video file: {:?}", file_path.as_ref());
            e
        })?;

        trace!("Selecting video stream from context");
        let stream_index = context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| anyhow!("No video stream found"))
            .map_err(|e| {
                error!("No video stream found in file: {:?}", file_path.as_ref());
                e
            })?
            .index();
        let stream = context
            .stream(stream_index)
            .ok_or_else(|| anyhow!("Failed to get video stream"))
            .map_err(|e| {
                error!(
                    "Failed to get video stream for stream index: {}",
                    stream_index
                );
                e
            })?;

        trace!("Getting decoder from stream parameters");
        let decoder = ffmpeg::codec::Context::from_parameters(stream.parameters())
            .map_err(|e| {
                error!("Failed to get codec from parameters");
                e
            })?
            .decoder()
            .video()
            .map_err(|e| {
                error!("Failed to get video decoder");
                e
            })?;

        debug!(
            "Video decoder initialized: codec = {:?}, width = {}, height = {}",
            decoder.id(),
            decoder.width(),
            decoder.height()
        );

        trace!("Initializing scaler for format conversion");
        let scaler = scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg_next::format::Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            scaling::Flags::BILINEAR,
        )
        .map_err(|e| {
            error!("Failed to construct scaler: {e}");
            e
        })?;

        trace!("VideoReader successfully initialized.");
        Ok(Self {
            context,
            decoder,
            stream_index,
            scaler,
            filename: file_path.as_ref().to_string_lossy().to_string(),
        })
    }

    pub fn total_frames(&self) -> Result<i64> {
        Ok(self
            .context
            .stream(self.stream_index)
            .ok_or_else(|| anyhow!("Stream {} is empty", self.stream_index))?
            .frames())
    }

    fn receive_frame(&mut self) -> Result<frame::Video> {
        let span = span!(Level::TRACE, "VideoReader::receive_frame");
        let _enter = span.enter();

        trace!("Receiving frame from decoder.");
        let mut decoded_frame = frame::Video::empty();
        self.decoder
            .receive_frame(&mut decoded_frame)
            .map_err(|e| {
                let errno: i32 = e.into();
                if errno == -EAGAIN {
                    trace!("Error receiving frame from decoder: {e}")
                } else {
                    error!("Error receiving frame from decoder: {e}");
                }

                e
            })?;

        debug!("Successfully received frame.");
        Ok(decoded_frame)
    }

    fn convert_frame_to_rgb(&mut self, frame: &frame::Video) -> Result<frame::Video> {
        let span = span!(Level::TRACE, "VideoReader::convert_frame_to_rgb");
        let _enter = span.enter();

        trace!("Converting frame to RGB.");
        let mut rgb_frame = frame::Video::empty();
        self.scaler.run(&frame, &mut rgb_frame).map_err(|e| {
            error!("Error converting frame to RGB: {:?}", e);
            e
        })?;

        debug!("Frame successfully converted to RGB.");
        Ok(rgb_frame)
    }

    pub fn next_nth_frame(&mut self, n: usize) -> Result<Option<frame::Video>> {
        let span = span!(Level::TRACE, "VideoReader::next_n_frame");
        let _enter = span.enter();

        if n == 0 {
            error!("Invalid argument: n must be greater than 0.");
            return Err(anyhow!("n must be greater than 0"));
        }

        debug!("Fetching the next {}th frame.", n);
        let mut skipped = 0;

        loop {
            if let Some((stream, packet)) = self.context.packets().next() {
                trace!("Processing packet from stream index: {}", stream.index());
                if stream.index() == self.stream_index {
                    self.decoder.send_packet(&packet).map_err(|e| {
                        error!("Error sending packet to decoder: {:?}", e);
                        e
                    })?;
                    while let Ok(frame) = self.receive_frame() {
                        skipped += 1;
                        debug!("Skipped frame count: {}", skipped);
                        if skipped == n {
                            let rgb_frame = self.convert_frame_to_rgb(&frame).map_err(|e| {
                                error!("Failed to convert frame to rgb: {e}");
                                e
                            })?;
                            debug!("Successfully fetched the next {}th frame.", n);
                            return Ok(Some(rgb_frame));
                        }
                    }
                }
            } else {
                debug!("No more packets available. Sending EOF to decoder.");
                self.decoder.send_eof().map_err(|e| {
                    error!("Error sending EOF to decoder: {:?}", e);
                    e
                })?;

                while let Ok(frame) = self.receive_frame() {
                    skipped += 1;
                    debug!("Skipped frame count after EOF: {}", skipped);
                    if skipped == n {
                        let rgb_frame = self.convert_frame_to_rgb(&frame).map_err(|e| {
                            error!("Failed to convert frame to rgb: {e}");
                            e
                        })?;
                        trace!("Successfully fetched the next {}th frame.", n);
                        return Ok(Some(rgb_frame));
                    }
                }
                break;
            }
        }

        warn!(
            "Failed to fetch the {}th frame: insufficient frames in the video.",
            n
        );
        Ok(None)
    }

    #[inline]
    pub fn next_frame(&mut self) -> Result<Option<frame::Video>> {
        self.next_nth_frame(1)
    }

    pub fn reset(&mut self) -> Result<()> {
        let span = span!(Level::TRACE, "VideoReader::reset");
        let _enter = span.enter();

        trace!("Resetting video context.");
        self.context = ffmpeg::format::input(&self.filename).map_err(|e| {
            error!("Failed to open video file: {:?}", self.filename);
            e
        })?;

        trace!("VideoReader successfully reset.");
        Ok(())
    }

    fn initialize_ffmpeg() {
        FFMPEG_INIT.call_once(|| {
            if let Err(e) = ffmpeg::init() {
                error!("Failed to initialize FFmpeg: {:?}", e);
                panic!("Failed to initialize FFmpeg");
            }
            trace!("FFmpeg library initialized successfully.");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Command;

    fn generate_test_video(output_path: &Path) {
        let status = Command::new("ffmpeg")
            .args(&[
                "-y", // 覆盖输出文件
                "-f",
                "lavfi", // 使用滤镜作为输入
                "-i",
                "color=c=red:s=128x128:d=5", // 纯红色，分辨率 128x128，持续 5 秒
                "-r",
                "25", // 帧率 25 fps
                "-pix_fmt",
                "yuv420p", // 像素格式
                "-loglevel",
                "error",
                output_path.to_str().unwrap(), // 输出路径
            ])
            .status()
            .expect("Failed to execute ffmpeg command");

        assert!(status.success(), "Failed to generate test video");
    }

    #[test]
    fn test_video_reader_initialization() {
        let temp_file = tempfile::Builder::new()
            .suffix(".mp4")
            .tempfile()
            .expect("Failed to create temporary file");
        generate_test_video(temp_file.path());

        let result = VideoReader::from_file(temp_file.path());
        assert!(
            result.is_ok(),
            "Failed to initialize VideoReader: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_total_frames() {
        let temp_file = tempfile::Builder::new()
            .suffix(".mp4")
            .tempfile()
            .expect("Failed to create temporary file");
        generate_test_video(temp_file.path());

        let video_reader =
            VideoReader::from_file(temp_file.path()).expect("Failed to initialize VideoReader");

        let total_frames = video_reader.total_frames();
        assert!(
            total_frames.is_ok(),
            "Failed to get total frames: {:?}",
            total_frames.err()
        );

        let frame_count = total_frames.unwrap();
        assert_eq!(
            frame_count, 125,
            "Expected 125 frames (5 seconds at 25 fps)"
        );
    }

    fn check_frame_data(frame: &frame::Video) {
        let frame_data = frame.data(0);
        for i in 0..frame_data.len() {
            if i % 3 == 0 {
                assert_eq!(frame_data[i], 253);
            } else {
                assert_eq!(frame_data[i], 0);
            }
        }
    }

    #[test]
    fn test_next_frame() {
        let temp_file = tempfile::Builder::new()
            .suffix(".mp4")
            .tempfile()
            .expect("Failed to create temporary file");
        generate_test_video(temp_file.path());

        let mut video_reader =
            VideoReader::from_file(temp_file.path()).expect("Failed to initialize VideoReader");

        let first_frame = video_reader.next_frame();
        assert!(
            first_frame.is_ok(),
            "Failed to read the next frame: {:?}",
            first_frame.err()
        );
        assert!(
            first_frame.as_ref().unwrap().is_some(),
            "Expected a frame, got None"
        );

        check_frame_data(&first_frame.unwrap().unwrap());
    }

    #[test]
    fn test_next_n_frame() {
        let temp_file = tempfile::Builder::new()
            .suffix(".mp4")
            .tempfile()
            .expect("Failed to create temporary file");
        generate_test_video(temp_file.path());

        let mut video_reader =
            VideoReader::from_file(temp_file.path()).expect("Failed to initialize VideoReader");

        let frame = video_reader.next_nth_frame(5);
        assert!(
            frame.is_ok(),
            "Failed to read the 5th frame: {:?}",
            frame.err()
        );
        assert!(
            frame.as_ref().unwrap().is_some(),
            "Expected the 5th frame, got None"
        );

        check_frame_data(&frame.unwrap().unwrap());
    }

    #[test]
    fn test_exceed_total_frames() {
        let temp_file = tempfile::Builder::new()
            .suffix(".mp4")
            .tempfile()
            .expect("Failed to create temporary file");
        generate_test_video(temp_file.path());

        let mut video_reader =
            VideoReader::from_file(temp_file.path()).expect("Failed to initialize VideoReader");

        let total_frames = video_reader
            .total_frames()
            .expect("Failed to get total frames");
        let result = video_reader.next_nth_frame((total_frames + 10) as usize); // 超过总帧数
        assert!(
            result.is_ok(),
            "Expected Ok(None) when exceeding total frames"
        );
        assert!(
            result.unwrap().is_none(),
            "Expected None when exceeding total frames"
        );
    }
}
