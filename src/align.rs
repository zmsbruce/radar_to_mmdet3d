use ffmpeg_next as ffmpeg;
use pyo3::{
    types::{PyAnyMethods, PyModule},
    Python,
};
use std::error::Error;

pub fn get_video_frame_num(filename: &str) -> Result<u32, Box<dyn Error>> {
    // 初始化 ffmpeg 库
    ffmpeg::init()?;

    // 打开视频文件
    let mut ictx = ffmpeg::format::input(&filename)?;

    // 查找视频流
    let video_stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?
        .index();

    // 初始化帧计数器
    let mut frame_cnt = 0;

    // 读取视频帧数据包
    for (stream, _) in ictx.packets() {
        if stream.index() == video_stream_index {
            frame_cnt += 1;
        }
    }

    Ok(frame_cnt)
}

pub fn get_rosbag_frame_num(
    filename: &str,
    lidar_topic: &str,
    video_topic: &str,
) -> Result<u32, Box<dyn Error>> {
    Python::with_gil(|py| {
        // 定义 Python 代码
        let code = r#"
import rosbag

def get_rosbag_frame_num(filename, lidar_topic, video_topic):
    bag = rosbag.Bag(filename)
    frame_count_list = []
    for topic in [lidar_topic, video_topic]:
        frame_count = 0
        for topic, msg, t in bag.read_messages(topics=[topic]):
            frame_count += 1
        frame_count_list.append(frame_count)
    bag.close()
    return min(frame_count_list)
"#;

        // 调用 Python 函数并传递参数
        let result: u32 = PyModule::from_code_bound(py, code, "", "")?
            .getattr("get_rosbag_frame_num")?
            .call1((filename, lidar_topic, video_topic))?
            .extract()?;

        // 返回结果
        Ok(result)
    })
}
