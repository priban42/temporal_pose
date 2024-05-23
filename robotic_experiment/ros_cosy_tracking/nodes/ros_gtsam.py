#!/usr/bin/env python
import sys
import rospy
import numpy as np
import threading
from cv_bridge import CvBridge
sys.path.insert(0, '/home/ros/sandbox_mf/gtsam_playground/scripts')
from SAM_dynamic_swap import SAM, SAMSettings
from ros_cosy_tracking.msg import CosyDetections, CosyDetection
from ros_cosy_tracking.msg import Track, Tracks
from geometry_msgs.msg import PoseStamped, Pose
from collections import defaultdict

from ros_cosy_tracking.buffer import Buffer
from ros_cosy_tracking.utils import SE3_2_posemsg, posemsg_2_SE3

bridge = CvBridge()

#  roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 serial_no:=105322250337
#  rosrun image_view image_view image:=/camera/color/image_raw


class Predictor:
    def __init__(self, 
                 topic_cosy: str, 
                 topic_kin: str,
                 topic_tracks: str,
                 topic_object_pose: str):
        rospy.init_node("predictor")
        sam_settings = SAMSettings(translation_dist_weight=0.5,
                                    mod=1,
                                    window_size=20,
                                    cov_drift_lin_vel=0.01,
                                    cov_drift_ang_vel=1.0,
                                    cov2_t=0.0000000001,
                                    cov2_R=0.0000000001,
                                    outlier_rejection_treshold=20,
                                    t_validity_treshold=0.725,
                                    R_validity_treshold=0.95,
                                    hysteresis_coef=1,
                                    velocity_prior_sigma=10)
        self.sam = SAM(settings = sam_settings)
        self.poses = {}
        self.state_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.pose_buffer = Buffer(1024, 2048)
        self.last_state = None
        self.last_solve_stamp = None
        rospy.Subscriber(topic_cosy, CosyDetections, callback=self.cb_cosy_detection, queue_size=1)
        rospy.Subscriber(topic_kin, PoseStamped, callback=self.cb_kinematics)
        self.tracks_pub = rospy.Publisher(topic_tracks, Tracks, queue_size=10)
        self.target_pose_pub = rospy.Publisher(topic_object_pose, PoseStamped, queue_size=10)
        self.counter = 0
        self.latched_track_id = None
        self.t0 = rospy.Time.now()

    @staticmethod
    def dets_to_lists(dets: list[CosyDetection]):
        T_cos = defaultdict(list)
        px_counts = defaultdict(list)
        for det in dets:
            T_co = posemsg_2_SE3(det.posecov.pose).homogeneous
            # Q = np.array(det.posecov.covariance).reshape((6, 6))
            obj_label = det.obj_label
            pixel_count = det.mask_pixel_count
            T_cos[obj_label].append(T_co)
            px_counts[obj_label].append(pixel_count)
        return T_cos, px_counts
    
    def cb_cosy_detection(self, msg: CosyDetections):
        # if self.counter < 30:
        print('D', (msg.header.stamp - self.t0).to_sec())

        # rospy.loginfo(f'cb_cosy_detection called')
        img_time_stamp = msg.header.stamp.to_sec()

        ts_querry, T_bc = self.pose_buffer.querry(img_time_stamp)

        if T_bc is not None:
            self.sam.insert_odometry_measurements()
            self.sam.insert_T_bc_detection(T_bc, img_time_stamp)

            all_T_cos, all_px_counts = Predictor.dets_to_lists(msg.dets)
            for obj_label in all_T_cos:
                T_cos = all_T_cos[obj_label]
                px_counts = all_px_counts[obj_label]
                self.sam.insert_T_co_detections(T_cos, obj_label, px_counts)
            self.sam.update_fls()
            with self.state_lock:
                self.last_state = self.sam.export_current_state()
                self.last_solve_stamp = img_time_stamp
                if self.counter > 30 and self.latched_track_id is None:
                    for obj_label in self.last_state:
                        for obj_instance in range(len(self.last_state[obj_label])):
                            if self.last_state[obj_label][obj_instance]['valid'] and obj_label == 'ycbv-obj_000002':
                                self.latched_track_id = self.last_state[obj_label][obj_instance]['id']

                                break
            print(self.counter, self.latched_track_id)
            self.counter += 1
        else:
            print('TOO SHORT')
            print('asked: ', img_time_stamp)
            print('first: ', self.pose_buffer.time_stamps[0])
            print('last: ', self.pose_buffer.time_stamps[-1])


    def cb_kinematics(self, msg: PoseStamped):
        # print(' K', (msg.header.stamp - self.t0).to_sec())

        T_bc = posemsg_2_SE3(msg.pose).homogeneous
        current_timestamp = msg.header.stamp

        latched_pose = None
        with self.pose_lock:
            self.pose_buffer.append(current_timestamp.to_sec(), T_bc)
        with self.state_lock:
            state = self.last_state
        if state is not None:
            tracks = Tracks()
            tracks.header.stamp = current_timestamp
            for obj_label in state:
                for obj_idx in range(len(state[obj_label])):
                    if state[obj_label][obj_idx]['valid'] == True:
                        entry = state[obj_label][obj_idx]
                        # rospy.loginfo(f'{obj_label},{obj_idx},{entry["id"]} is valid')
                        last_T_bo = entry['T_bo']
                        if 'nu' in entry:
                            nu = entry['nu']
                        else:
                            nu = np.zeros(6)
                        dt = current_timestamp.to_sec() - self.last_solve_stamp
                        T_bo = self.sam.extrapolate_T_bo(last_T_bo, nu, dt)
                        if entry['id'] == self.latched_track_id:
                            latched_pose = T_bo
                        track = Track()
                        track.label = obj_label
                        track.track_id = entry['id']
                        track.pose = SE3_2_posemsg(T_bo)
                        tracks.tracks.append(track)
            if latched_pose is not None:
                pose_st_msg = PoseStamped()
                pose_st_msg.header = msg.header
                pose_st_msg.pose = SE3_2_posemsg(latched_pose)
                self.target_pose_pub.publish(pose_st_msg)
            # else:
                # print(f"latched_track_id:{self.latched_track_id}")
            self.tracks_pub.publish(tracks)
        else:
            rospy.loginfo(f'state is None')


if __name__ == "__main__":

    topic_cosy = 'cosy_detections'
    topic_kin = 'pose_camera'
    topic_tracks = 'tracks'
    topic_object_pose = 'target_object_pose'
    Predictor(topic_cosy, topic_kin, topic_tracks, topic_object_pose)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass