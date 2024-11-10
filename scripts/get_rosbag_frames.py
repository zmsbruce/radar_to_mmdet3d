import rosbag
import rospy


def get_rosbag_frame_num(filename, lidar_topic):
    try:
        rospy.init_node('vidbag2mmd', anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    bag = rosbag.Bag(filename)

    frame_count = 0
    for _, _, _ in bag.read_messages(topics=[lidar_topic]):
        frame_count += 1
        
    bag.close()
    return frame_count
