#!/usr/bin/env python

import pickle
from pathlib import Path
import numpy as np
import pinocchio as pin

import rospy
import rospkg

from ros_cosy_tracking.msg import CosyDetection, CosyDetections
from geometry_msgs.msg import PoseWithCovariance

"""
Publish fake cosypose + covariance results. Files stored in data/ folder
"""

RATE = 1  # Hz

# Get path to data folder
rospack = rospkg.RosPack()
data_dir = Path(rospack.get_path('ros_cosy_tracking')) / 'data'

preds_file = 'frames_prediction.p'
pix_file = 'frames_px_counts.p'



with open(data_dir / preds_file, 'rb') as fp:
    preds = pickle.load(fp)

with open(data_dir / pix_file, 'rb') as fp:
    pix_counts = pickle.load(fp)


pub = rospy.Publisher('cosydetections', CosyDetections, queue_size=10)
rospy.init_node('ros_cosycov_simulation', anonymous=True)
rate = rospy.Rate(RATE) # 10hz
k = 0
while not rospy.is_shutdown():
    t = rospy.Time.now() - rospy.Time.from_sec(2)
    rospy.loginfo(f'Parsing pose {k}, {t}')

    hello_str = "hello world %s" % rospy.get_time()
    preds_k, pix_counts_k = preds[k], pix_counts[k]

    if not len(preds_k) == len(pix_counts_k):
        import ipdb; ipdb.set_trace()
        hello_str = "BAD"

    cosy_dets = CosyDetections()
    cosy_dets.header.seq = k 
    cosy_dets.header.stamp = t
    cosy_dets.header.frame_id = 'camera_dummy'
    for label in preds_k:
        for T, pix_count in zip(preds_k[label], pix_counts_k[label]): 

            T = pin.SE3(T)
            xyz_qxyzw = pin.SE3ToXYZQUAT(T)

            det = CosyDetection()
            det.posecov = PoseWithCovariance()
            det.posecov.pose.position.x = xyz_qxyzw[0]
            det.posecov.pose.position.y = xyz_qxyzw[1]
            det.posecov.pose.position.z = xyz_qxyzw[2]
            det.posecov.pose.orientation.x = xyz_qxyzw[3]
            det.posecov.pose.orientation.y = xyz_qxyzw[4]
            det.posecov.pose.orientation.z = xyz_qxyzw[5]
            det.posecov.pose.orientation.w = xyz_qxyzw[6]

            det.obj_label = label
            det.score = 0.942
            det.mask_pixel_count = pix_count

            cov = np.eye(6)
            det.posecov.covariance = cov.flatten().tolist()

            cosy_dets.dets.append(det)


    pub.publish(cosy_dets)
    rate.sleep()

    k += 1



