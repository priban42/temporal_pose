#!/usr/bin/env python

"""
Reads object poses in camera frame + robot kinematics, outputs 1 object pose in robot frame. 
"""

import rospy
from geometry_msgs.msg import PoseStamped

from ros_cosy_tracking.msg import CosyDetections
from ros_cosy_tracking.utils import posemsg_2_SE3, SE3_2_posemsg
from ros_cosy_tracking.buffer import Buffer


class CosySelection:

    def __init__(self, 
                 topic_cosy: str,
                 topic_object_pose: str,
                 target_object_label: str
                 ) -> None:
        rospy.init_node("cosy_selection")
        
        self.target_object_label = target_object_label
        rospy.Subscriber(topic_cosy, CosyDetections, queue_size=1)
        self.pose_obj_pub = rospy.Publisher(topic_object_pose, PoseStamped, queue_size=1)

        self.pose_buffer = Buffer()

    def cb_cosydets(self, msg: CosyDetections):
        """
        Reads kinematic buffer, get closest camera pose measurement, transform target pose to robot frame.
        """
        ts = msg.header.stamp.to_sec()
        for det in msg.dets:
            if det.label == self.target_obj_label:
                T_co = posemsg_2_SE3(det.posecov.pose)
                ts_q, T_bc = self.pose_buffer.querry(ts)
                if T_bc is not None:
                    T_bo = T_bc * T_co
                    msg_obj = PoseStamped()
                    msg_obj.header = msg.header
                    msg_obj.pose = SE3_2_posemsg(T_bo)
                    self.pose_obj_pub.publish(msg_obj)

    def cb_kine(self, msg: PoseStamped):
        ts = msg.header.stamp.to_sec()
        T_bc = posemsg_2_SE3(msg.pose)
        self.pose_buffer.append(ts, T_bc)


if __name__ == '__main__':
    topic_cosy = 'cosy_detections'
    topic_object_pose = 'target_object_pose'
    target_object_label = 'todo'

    CosySelection(topic_cosy, topic_object_pose, target_object_label)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass