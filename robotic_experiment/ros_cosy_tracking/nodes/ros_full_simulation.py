#!/usr/bin/env python

import pickle
import json
from pathlib import Path
import numpy as np
import pinocchio as pin

import rospy
import rospkg


from ros_cosy_tracking.msg import CosyDetection, CosyDetections
from geometry_msgs.msg import PoseWithCovariance, Pose

"""
Publish fake cosypose + covariance results. Files stored in data/ folder
"""

def load_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_scene_camera(path):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = {}
        entry["cam_K"] = np.array(data[str(i+1)]["cam_K"]).reshape((3, 3))
        T_cw = np.zeros((4, 4))
        T_cw[:3, :3] = np.array(data[str(i+1)]["cam_R_w2c"]).reshape((3, 3))
        T_cw[:3, 3] = np.array(data[str(i+1)]["cam_t_w2c"])/1000
        T_cw[3, 3] = 1
        entry["T_cw"] = T_cw
        parsed_data.append(entry)
    return parsed_data


def SE3_to_ros_pose(T):
    T_pin = pin.SE3(T)
    xyz_qxyzw = pin.SE3ToXYZQUAT(T_pin)
    pose = Pose()
    pose.position.x = xyz_qxyzw[0]
    pose.position.y = xyz_qxyzw[1]
    pose.position.z = xyz_qxyzw[2]
    pose.orientation.x = xyz_qxyzw[3]
    pose.orientation.y = xyz_qxyzw[4]
    pose.orientation.z = xyz_qxyzw[5]
    pose.orientation.w = xyz_qxyzw[6]
    return pose

def entry_to_cosy_detections(all_predictions, all_pix_counts, frame, tima_stamp):
    preds = all_predictions[frame]
    pix_counts = all_pix_counts[frame]
    cosy_dets = CosyDetections()
    cosy_dets.header.seq = frame
    cosy_dets.header.stamp = tima_stamp - rospy.Time.from_sec(0.2)
    cosy_dets.header.frame_id = 'camera_dummy'
    for label in preds:
        for T, pix_count in zip(preds[label], pix_counts[label]):

            T = pin.SE3(T)
            det = CosyDetection()
            det.posecov = PoseWithCovariance()
            det.posecov.pose = SE3_to_ros_pose(T)
            det.obj_label = label
            det.score = 0.942
            det.mask_pixel_count = pix_count

            cov = np.eye(6)
            det.posecov.covariance = cov.flatten().tolist()

            cosy_dets.dets.append(det)
    return cosy_dets

def entry_to_fk(all_camera_poses, frame, time_stamp):
    T_cw = all_camera_poses[frame]['T_cw']
    T_wc = pin.SE3(T_cw).inverse()
    camera_pose_msg = SE3_to_ros_pose(T_wc)
    return camera_pose_msg

def main():
    rospack = rospkg.RosPack()
    data_dir = Path(rospack.get_path('ros_cosy_tracking')) / 'data'
    preds = load_pickle(data_dir / 'frames_prediction.p')
    pix_counts = load_pickle(data_dir / 'frames_px_counts.p')
    camera_poses = load_scene_camera(data_dir/'scene_camera.json')

    RATE = 10 # Hz

    cosy_pub = rospy.Publisher('cosydetections', CosyDetections, queue_size=5)
    fk_pub = rospy.Publisher('camerapose', Pose, queue_size=10)
    rospy.init_node('ros_cosycov_simulation', anonymous=True)
    rate = rospy.Rate(RATE)  # 10hz
    frame = 0
    k = 0
    while not rospy.is_shutdown():
        t = rospy.Time.now()
        # rospy.loginfo(f'Parsing pose {frame}, {t}')

        # hello_str = "hello world %s" % rospy.get_time()
        fk_msg = entry_to_fk(camera_poses, frame, t)
        fk_pub.publish(fk_msg)
        if k % 10 == 0:
            cosy_dets_msg = entry_to_cosy_detections(preds, pix_counts, frame, t)
            cosy_pub.publish(cosy_dets_msg)
            frame += 1


        k += 1
        if frame >= len(preds):
            frame = 0
        rate.sleep()

if __name__ == "__main__":
    main()
