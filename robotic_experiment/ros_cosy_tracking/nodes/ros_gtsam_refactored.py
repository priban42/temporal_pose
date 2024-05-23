#!/usr/bin/env python
import sys
import rospy
import numpy as np
import threading
from cv_bridge import CvBridge
sys.path.insert(0, '/home/ros/sandbox_mf/gtsam_playground/scripts/refactored')
from SamWrapper import SamWrapper
from State import State, BareTrack
import GlobalParams# import GlobalParams, rapid_tracking, robust_tracking
import cov as cov

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
        self.params = GlobalParams.GlobalParams(
                                cov_drift_lin_vel=0.002,
                                cov_drift_ang_vel=0.02,
                                outlier_rejection_treshold=0.2,
                                t_validity_treshold=6.4e-4,
                                R_validity_treshold=0.08,
                                max_track_age=2.0,
                                # t_validity_treshold=1e3,
                                # R_validity_treshold=1e3,
                                max_derivative_order=1,
                                reject_overlaps=0.2)
        # params = GlobalParams.robust_tracking
        # params = GlobalParams.rapid_tracking
        self.sam:SamWrapper = SamWrapper(self.params)
        self.state = None
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
        ret = defaultdict(list)
        for det in dets:
            T_co = posemsg_2_SE3(det.posecov.pose).homogeneous
            # Q = np.array(det.posecov.covariance).reshape((6, 6))
            obj_label = det.obj_label
            pixel_count = det.mask_pixel_count
            Q = cov.measurement_covariance(T_co, pixel_count)
            ret[obj_label].append({"T_co":T_co, "Q":Q})
        return ret
    
    def cb_cosy_detection(self, msg: CosyDetections):
        # if self.counter < 30:
        # print('D', (msg.header.stamp - self.t0).to_sec())

        # rospy.loginfo(f'cb_cosy_detection called')
        img_time_stamp = msg.header.stamp.to_sec()

        ts_querry, T_wc = self.pose_buffer.querry(img_time_stamp)

        if T_wc is not None:
            Q_T_wc = np.eye(6) * 10 ** (-6)

            detections = Predictor.dets_to_lists(msg.dets)
            self.sam.insert_detections({"T_wc":T_wc, "Q":Q_T_wc}, detections, img_time_stamp)
            self.state = self.sam.get_state()
            # with self.state_lock:
            #     self.last_state = self.sam.export_current_state()
            #     self.last_solve_stamp = img_time_stamp
            if self.counter > 30 and self.latched_track_id is None:
                for obj_label in self.state.bare_tracks:
                    for obj_instance in range(len(self.state.bare_tracks[obj_label])):
                        if obj_label == 'ycbv-obj_000002':
                            Q = self.state.bare_tracks[obj_label][obj_instance].Q
                            valid = State.is_valid(Q, self.params.t_validity_treshold, self.params.R_validity_treshold)
                            if valid:
                                self.latched_track_id = self.state.bare_tracks[obj_label][obj_instance].idx
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
        # print(state)
        if self.state is not None:
            extrapolated_state = self.state.get_extrapolated_state(current_timestamp.to_sec(), T_bc)
            tracks = Tracks()
            tracks.header.stamp = current_timestamp
            for obj_label in extrapolated_state:
                for obj_idx in range(len(extrapolated_state[obj_label])):
                    if extrapolated_state[obj_label][obj_idx]['valid'] == True:
                        entry = extrapolated_state[obj_label][obj_idx]
                        T_wo = entry["T_wo"]

                        if entry['id'] == self.latched_track_id:
                            latched_pose = T_wo
                        track = Track()
                        track.label = obj_label
                        track.track_id = entry['id']
                        track.pose = SE3_2_posemsg(T_wo)
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