import rosbag


def get_rosbag_frame_num(filename, lidar_topic):
    bag = rosbag.Bag(filename)

    frame_count = 0
    for _, _, _ in bag.read_messages(topics=[lidar_topic]):
        frame_count += 1
        
    bag.close()
    return frame_count
