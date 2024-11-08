import rosbag


def get_rosbag_frame_num(filename, lidar_topic, video_topic):
    bag = rosbag.Bag(filename)
    frame_count_list = []
    for topic in [lidar_topic, video_topic]:
        frame_count = 0
        for topic, _, _ in bag.read_messages(topics=[topic]):
            frame_count += 1
        frame_count_list.append(frame_count)
    bag.close()
    return min(frame_count_list)
