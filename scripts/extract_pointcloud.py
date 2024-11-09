import rosbag
import numpy as np
import os
import logging
import sensor_msgs.point_cloud2 as pc2


# 函数：处理每一帧的点云数据
def process_pointclouds(bag_file, pointcloud_topic, output_dir):

    # 配置 logging
    logging.basicConfig(level=logging.INFO,  # 设置日志级别
                        # 日志格式
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')  # 时间格式

    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    # 打开 rosbag 文件
    bag = rosbag.Bag(bag_file)

    frame_count = 0  # 计数处理的点云帧数

    # 读取指定话题中的点云数据
    for topic, msg, t in bag.read_messages(topics=[pointcloud_topic]):
        frame_count += 1
        logging.info(f"Processing frame {frame_count}: timestamp {t}")

        # 将 PointCloud2 消息转换为点云数据
        point_cloud = pc2.read_points(msg, field_names=(
            "x", "y", "z", "intensity"), skip_nans=True)

        # 将点云数据转换为 NumPy 数组
        points = np.array(list(point_cloud))

        # 检查点云是否为空
        if points.size == 0:
            logging.warning(f"Frame {frame_count} is empty. Skipping...")
            continue

        # 定义 .bin 文件的输出路径
        bin_file = os.path.join(output_dir, f'frame_{frame_count:06d}.bin')

        # 保存点云为 .bin 文件
        points.astype(np.float32).tofile(bin_file)
        logging.info(f"Saved point cloud to {bin_file}")

    # 关闭 rosbag 文件
    bag.close()
    logging.info(f"Finished processing {frame_count} frames.")
