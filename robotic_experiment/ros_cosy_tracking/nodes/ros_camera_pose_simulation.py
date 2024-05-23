#!/usr/bin/env python

import pickle
from pathlib import Path
import numpy as np
import pinocchio as pin

import rospy
import rospkg

from ros_cosy_tracking.msg import CosyDetection, CosyDetections
from geometry_msgs.msg import PoseWithCovariance, Pose

from ros_cosy_tracking.utils import SE3_2_posemsg

"""
Publish fake cosypose + covariance results. Files stored in data/ folder
"""

RATE = 10  # Hz

# Get path to data folder
rospack = rospkg.RosPack()
data_dir = Path(rospack.get_path('ros_cosy_tracking')) / 'data'


pub = rospy.Publisher('camerapose', Pose, queue_size=10)
rospy.init_node('ros_camera_pose_simulation', anonymous=True)
rate = rospy.Rate(RATE) # 10hz
k = 0
while not rospy.is_shutdown():
    t = rospy.Time.now()
    rospy.loginfo(f'Parsing pose {k}, {t}')

    T_bc = np.eye(4)

    pose = SE3_2_posemsg(T_bc)

    pub.publish(pose)
    rate.sleep()
    k += 1



