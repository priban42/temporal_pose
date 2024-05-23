#!/usr/bin/env python

import numpy as np
import pinocchio as pin

import rospy
from geometry_msgs.msg import PoseStamped

from ros_cosy_tracking.utils import posemsg_2_SE3, SE3_2_posemsg


SIM_MODES = {
    'SINUSOID', 
    'TRIANG', 
}

# MODE = 'SINUSOID'
MODE = 'TRIANG'

def triang(t, f):
    # https://en.wikipedia.org/wiki/Triangle_wave
    # with f = 1/p
    return 2*np.abs(t*f - np.floor(t*f + 0.5))


class SimulateObject:

    def __init__(self, 
                 topic_object_pose: str, 
                 topic_kin: str, 
                 T_co_init: pin.SE3,
                 rate_poses: int) -> None:
        rospy.init_node('simulate_tracks')

        self.T_co_init = T_co_init
        # self.tracks_pub = rospy.Publisher(topic_object_tracks, Tracks, queue_size=1)
        self.pose_obj_pub = rospy.Publisher(topic_object_pose, PoseStamped, queue_size=1)

        self.first_kin_received = False
        self.T_bo_init = None

        self.t0 = None
        self.t_prev = None

        # SINUSOID
        self.freq = np.array([0.2,0,0])
        self.omg = 2*np.pi*self.freq
        self.amp = np.array([0.3,0,0])

        rospy.Subscriber(topic_kin, PoseStamped, self.cb_sub_kin)
        rospy.Timer(rospy.Duration(1.0/rate_poses), self.cb_timer_object)

    def cb_sub_kin(self, msg: PoseStamped):
        if not self.first_kin_received:
            T_b_c = posemsg_2_SE3(msg.pose)
            T_bc_init = T_b_c.copy()
            self.first_kin_received = True
            self.T_bo_init = T_bc_init * self.T_co_init

    def cb_timer_object(self, event=None):
        # get position of object(s) in camera frame, send through publisher
        if MODE not in SIM_MODES:
            print('BAD mode ', MODE)
            return
        t = rospy.Time.now()
        if self.t_prev is None:
            self.t_prev = t
            return

        if self.T_bo_init is not None:
            if self.t0 is None:
                self.t0 = t
                self.t_last_switch = t

            delta_t = (t - self.t0).to_sec()
            if MODE == 'SINUSOID':
                delta_trans = self.amp * np.sin(self.omg*delta_t)  # SINUSOID
            elif MODE == 'TRIANG':
                delta_trans = self.amp * triang(delta_t, self.freq)  # CST VEL

            T_bo = self.T_bo_init.copy()
            T_bo.translation = T_bo.translation + delta_trans

            msg = PoseStamped()
            msg.pose = SE3_2_posemsg(T_bo)
            self.pose_obj_pub.publish(msg)

            print(msg.pose)

        self.t_prev = t

if __name__ == '__main__':
    topic_kin = 'pose_camera'
    topic_object_pose = 'target_object_pose'

    RATE_POSES = 10

    T_co_init = pin.SE3.Identity()
    T_co_init.translation[2] = 0.1

    node = SimulateObject(topic_object_pose, topic_kin, T_co_init, RATE_POSES)

    try:
        rospy.spin()
    except (rospy.ROSInterruptException, rospy.ROSInitException):
        pass
