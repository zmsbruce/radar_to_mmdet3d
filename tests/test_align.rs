use std::process::Command;
use std::{error::Error, path::PathBuf};
use tempfile::tempdir;

use vidbag2mmd::align::{get_rosbag_frame_num, get_video_frame_num};

#[test]
fn test_get_video_frame_num() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let video_path = dir.path().join("test_video.mp4");

    create_test_video(&video_path)?;

    let frame_count = get_video_frame_num(video_path.to_str().ok_or(format!(
        "Failed to convert video path {:?} to string",
        video_path
    ))?)?;
    assert!(frame_count == 30, "Frame count should be equal to 30");

    Ok(())
}

#[test]
fn test_get_rosbag_frame_num() -> Result<(), Box<dyn Error>> {
    let filename = "tests/test.bag"; // A bag with 10 frames of /lidar and 8 frames of /camera
    let lidar_topic = "/lidar";
    let video_topic = "/camera";

    let frame_count = get_rosbag_frame_num(filename, lidar_topic, video_topic)?;
    assert_eq!(frame_count, 8, "Frame count should be 10");

    Ok(())
}

fn create_test_video(output_path: &PathBuf) -> Result<(), std::io::Error> {
    // 使用 ffmpeg 命令生成一个 1 秒的 30 帧视频，大小为 128x128
    Command::new("ffmpeg")
        .args(&[
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=128x128:rate=30",
            output_path.to_str().unwrap(),
            "-y",
        ])
        .output()?;

    assert!(output_path.exists(), "Test video was not created");

    Ok(())
}
