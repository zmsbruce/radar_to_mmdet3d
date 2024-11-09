use ffmpeg_next::{
    self as ffmpeg,
    format::Pixel,
    frame::Video,
    software::scaling::{Context, Flags},
};
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use pyo3::{
    types::{PyAnyMethods, PyModule},
    Python,
};
use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    fs::{self, File},
    io::{AsyncWriteExt, BufWriter},
    sync::Semaphore,
    task,
};

pub struct Aligner {
    left_camera_video_filename: String,
    middle_camera_video_filename: String,
    right_camera_video_filename: String,
    lidar_rosbag_filename: String,
    lidar_rosbag_topic: String,
}

impl Aligner {
    pub fn new(
        left_camera_video_filename: &str,
        middle_camera_video_filename: &str,
        right_camera_video_filename: &str,
        lidar_rosbag_filename: &str,
        lidar_rosbag_topic: &str,
    ) -> Self {
        Self {
            left_camera_video_filename: left_camera_video_filename.to_string(),
            middle_camera_video_filename: middle_camera_video_filename.to_string(),
            right_camera_video_filename: right_camera_video_filename.to_string(),
            lidar_rosbag_filename: lidar_rosbag_filename.to_string(),
            lidar_rosbag_topic: lidar_rosbag_topic.to_string(),
        }
    }

    pub async fn align_frames(&self, output_dir: &str) -> Result<(), Box<dyn Error>> {
        // 1. 获取 LIDAR 数据的帧数
        log::info!("Getting LIDAR frame count from ROS bag...");
        let lidar_frame_count = self.get_rosbag_frame_num()?;
        log::info!("LIDAR frame count: {}", lidar_frame_count);

        // 2. 为每个视频提取与 LIDAR 对齐的帧数
        log::info!("Extracting frames from left camera...");
        Self::extract_frames(
            &self.left_camera_video_filename,
            &format!("{}/left", output_dir),
            lidar_frame_count as usize,
        )
        .await?;

        log::info!("Extracting frames from middle camera...");
        Self::extract_frames(
            &self.middle_camera_video_filename,
            &format!("{}/middle", output_dir),
            lidar_frame_count as usize,
        )
        .await?;

        log::info!("Extracting frames from right camera...");
        Self::extract_frames(
            &self.right_camera_video_filename,
            &format!("{}/right", output_dir),
            lidar_frame_count as usize,
        )
        .await?;

        log::info!("All frames extracted and aligned successfully.");

        Ok(())
    }

    fn get_rosbag_frame_num(&self) -> Result<u32, Box<dyn Error>> {
        log::info!("Starting to get ROS bag frame count...");

        Python::with_gil(|py| {
            // 定义 Python 代码
            log::debug!("Loading Python script for ROS bag frame extraction...");
            let code = include_str!("../scripts/get_rosbag_frames.py");

            // 调用 Python 函数并传递参数
            log::debug!(
                "Calling Python function 'get_rosbag_frame_num' with ROS bag filename: '{}' and topic: '{}'",
                self.lidar_rosbag_filename,
                self.lidar_rosbag_topic
            );
            let result: u32 = PyModule::from_code_bound(py, code, "", "")?
                .getattr("get_rosbag_frame_num")?
                .call1((&self.lidar_rosbag_filename, &self.lidar_rosbag_topic))?
                .extract()?;

            // 返回结果
            Ok(result)
        })
    }

    pub fn convert_rosbag_to_clouds_file(&self, output_dir: &str) -> Result<(), Box<dyn Error>> {
        log::info!("Starting to convert rosbag to files if point cloud...");

        Python::with_gil(|py| {
            // 定义 Python 代码
            log::debug!("Loading Python script for converting rosbag to files if point cloud...");
            let code = include_str!("../scripts/extract_pointclouds.py");

            // 调用 Python 函数并传递参数
            log::debug!(
                "Calling Python function 'extract_pointclouds' with bag filename: '{}', topic: '{}' and output directory: '{}'",
                &self.lidar_rosbag_filename,
                &self.lidar_rosbag_topic,
                output_dir
            );
            PyModule::from_code_bound(py, code, "", "")?
                .getattr("extract_pointclouds")?
                .call1((
                    &self.lidar_rosbag_filename,
                    &self.lidar_rosbag_topic,
                    output_dir,
                ))?;

            Ok(())
        })
    }

    pub async fn extract_frames(
        video_path: &str,
        output_dir: &str,
        target_frames: usize,
    ) -> Result<(), Box<dyn Error>> {
        // 初始化日志
        log::info!("Starting frame extraction from video: {}", video_path);
        log::debug!(
            "Output directory: {}, Target frames: {}",
            output_dir,
            target_frames
        );

        // 打开视频文件
        log::debug!("Opening video file: {}", video_path);
        let mut context = ffmpeg::format::input(&video_path)?;

        // 查找视频流
        let video_stream_index = context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("Could not find video stream")?
            .index();
        log::info!("Found video stream with index: {}", video_stream_index);

        // 获取视频的总帧数
        let video_stream = context.stream(video_stream_index).unwrap();
        let total_frames = video_stream.frames();
        log::info!("Total frames in video: {}", total_frames);

        // 如果视频帧数少于目标帧数，返回错误
        if total_frames < target_frames as i64 {
            log::error!(
                "Video has fewer frames ({}) than the target frame count ({})",
                total_frames,
                target_frames
            );
            return Err(format!(
                "Video has fewer frames ({}) than the target frame count ({})",
                total_frames, target_frames
            )
            .into());
        }

        // 计算抽帧步长
        let step = total_frames as f64 / target_frames as f64;
        log::debug!("Frame extraction step calculated: {}", step);

        // 获取解码器
        log::debug!("Initializing video decoder...");
        let mut decoder =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?
                .decoder()
                .video()?;
        log::info!("Video decoder initialized.");

        // 确保输出目录存在，异步创建目录
        log::debug!("Ensuring output directory exists: {}", output_dir);
        let output_dir_path = Path::new(output_dir);
        fs::create_dir_all(output_dir_path).await?;

        // 共享输出目录路径
        let output_dir_path = Arc::new(output_dir_path.to_path_buf());

        // 使用信号量控制并发，假设最多同时保存 10 个帧
        let semaphore = Arc::new(Semaphore::new(10));

        // 创建一个向量来存储所有的 JoinHandle
        let mut handles = Vec::new();

        // 解码帧并异步保存
        let mut frame_index = 0;
        let mut extracted_frames = 0;
        let mut next_frame_to_extract = 0.0;

        log::info!("Starting frame extraction loop...");
        for (stream, packet) in context.packets() {
            if stream.index() == video_stream_index {
                log::trace!("Processing packet from video stream...");
                decoder.send_packet(&packet)?;

                let mut frame = ffmpeg::frame::Video::empty();
                while let Ok(()) = decoder.receive_frame(&mut frame) {
                    log::trace!("Received a new frame: index {}", frame_index);

                    // 检查是否应该抽取这一帧
                    if (frame_index as f64) >= next_frame_to_extract {
                        log::debug!("Extracting frame {}...", frame_index);

                        // 构建输出路径
                        let output_path =
                            output_dir_path.join(format!("frame_{:07}.png", extracted_frames));
                        let frame_copy = frame.clone(); // 复制帧，因为它会被复用
                        let output_path = output_path.clone();
                        let permit = Arc::clone(&semaphore).acquire_owned().await.unwrap();

                        // 异步保存帧
                        let handle = task::spawn(async move {
                            let _permit = permit; // 保持 permit 的生命周期，任务完成后释放
                            match Self::save_frame_as_png(&frame_copy, &output_path).await {
                                Ok(_) => log::info!("Saved frame {} successfully", frame_index),
                                Err(e) => {
                                    log::error!("Failed to save frame {}: {:?}", frame_index, e)
                                }
                            }
                        });

                        handles.push(handle);

                        extracted_frames += 1;
                        next_frame_to_extract += step;

                        // 如果已经提取了足够的帧，则停止
                        if extracted_frames >= target_frames {
                            break;
                        }
                    }

                    frame_index += 1;
                }
            }

            // 如果已经提取了足够的帧，则停止
            if extracted_frames >= target_frames {
                break;
            }
        }

        // 等待所有的保存任务完成
        for handle in handles {
            if let Err(e) = handle.await {
                log::error!("A background task failed: {:?}", e);
            }
        }

        log::info!("Frame extraction process completed successfully.");
        Ok(())
    }

    async fn save_frame_as_png(frame: &Video, output_path: &PathBuf) -> Result<(), Box<dyn Error>> {
        // 获取帧的宽度和高度
        let width = frame.width();
        let height = frame.height();
        log::debug!("Frame dimensions: {}x{}", width, height);

        // 检查帧的像素格式
        let source_format = frame.format();
        let target_format = Pixel::RGB24;
        log::debug!("Source pixel format: {:?}", source_format);

        // 如果帧的格式不是 RGB24，则进行格式转换
        let frame_to_use = if source_format != target_format {
            log::debug!(
                "Converting frame format from {:?} to {:?}",
                source_format,
                target_format
            );

            // 创建一个用于格式转换的上下文
            let mut scaler = Context::get(
                source_format,
                width,
                height,
                target_format,
                width,
                height,
                Flags::BILINEAR,
            )?;

            // 创建一个新的空的 RGB24 格式的帧，并执行格式转换
            let mut rgb_frame = Video::empty();
            scaler.run(frame, &mut rgb_frame)?;
            log::info!("Frame format conversion successful.");

            rgb_frame
        } else {
            log::debug!("Frame is already in RGB24 format, no conversion needed.");

            // 如果帧已经是 RGB24 格式，直接使用原始帧
            frame.clone()
        };

        // 分配缓冲区，假设是 RGB24 格式
        let buffer_size = (width * height * 3) as usize;
        log::debug!("Allocating buffer of size: {} bytes", buffer_size);
        let mut buf = vec![0u8; buffer_size];

        // 拷贝帧数据到缓冲区
        log::trace!("Copying frame data into buffer...");
        frame_to_use
            .data(0)
            .iter()
            .enumerate()
            .for_each(|(i, &val)| buf[i] = val);
        log::debug!("Frame data copied to buffer.");

        // 创建 PNG 编码器并将图像数据编码为 PNG 格式
        log::debug!("Creating PNG encoder...");
        let mut png_data = Vec::new();
        {
            let encoder = PngEncoder::new(&mut png_data);
            log::debug!("Encoding frame data as PNG...");
            encoder.write_image(&buf, width as u32, height as u32, ColorType::Rgb8.into())?;
        }
        log::info!("Frame encoded as PNG, size: {} bytes", png_data.len());

        // 使用 tokio 异步文件 I/O 打开文件
        log::debug!("Creating output file at: {:?}", output_path);
        let file = File::create(output_path).await?;
        let mut writer = BufWriter::new(file);

        // 异步写入 PNG 数据
        log::debug!("Writing PNG data to file...");
        writer.write_all(&png_data).await?;
        log::debug!("PNG data written to file.");

        // 确保数据刷新到磁盘
        log::debug!("Flushing file buffer...");
        writer.flush().await?;
        log::info!("PNG file successfully saved at: {:?}", output_path);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, fs, process::Command};
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_get_rosbag_frame_num() -> Result<(), Box<dyn Error>> {
        let filename = "assets/test/test.bag"; // A bag with 10 frames of /lidar and 8 frames of /camera
        let lidar_topic = "/lidar";

        let aligner = Aligner::new("", "", "", &filename, &lidar_topic);
        let frame_count = aligner.get_rosbag_frame_num()?;
        assert_eq!(frame_count, 10, "Frame count should be 10");

        Ok(())
    }

    #[test]
    fn test_get_rosbag_frame_num_nonexistent_path() -> Result<(), Box<dyn Error>> {
        let nonexistent_bag_path = "nonexistent.bag";
        let lidar_topic = "/lidar";

        let aligner = Aligner::new("", "", "", &nonexistent_bag_path, &lidar_topic);
        let result = aligner.get_rosbag_frame_num();
        assert!(
            result.is_err(),
            "The function should return an error when the ROS bag file does not exist"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_extract_frames() -> Result<(), Box<dyn Error>> {
        // 创建临时目录用于保存视频文件
        let temp_dir = tempdir()?;
        let video_path = temp_dir.path().join("test_video.mp4");

        // 使用 ffmpeg 生成测试视频
        let output = Command::new("ffmpeg")
            .args(&[
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=5:size=1280x720:rate=30",
                video_path.to_str().unwrap(),
            ])
            .output()?;

        // 检查 ffmpeg 命令是否成功执行
        if !output.status.success() {
            // 使用 `into()` 将错误字符串转换为 `Box<dyn Error>`
            return Err(format!("Failed to generate video: {:?}", output).into());
        }

        // 检查生成的视频文件是否存在
        assert!(video_path.exists(), "Video file should exist");

        // 创建临时目录用于保存帧
        let output_temp_dir = tempdir()?; // 使用一个不同的目录生成帧
        let output_dir = output_temp_dir.path().to_str().unwrap();

        // 提取 13 帧
        Aligner::extract_frames(
            video_path
                .to_str()
                .ok_or_else(|| format!("Failed to convert {:#?} to string", video_path))?, // 错误处理
            output_dir,
            13,
        )
        .await?;

        // 检查输出目录是否生成了 13 个帧文件
        let frame_files: Vec<_> = fs::read_dir(output_dir)?
            .filter_map(|entry| entry.ok())
            .collect();
        assert_eq!(frame_files.len(), 13, "Should have extracted 13 frames");

        Ok(())
    }
}
