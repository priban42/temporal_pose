#!/usr/bin/env python

import os
import time
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

import panda_py
from panda_py import libfranka, controllers

# panda-py is chatty, activate information log level
import logging
logging.basicConfig(level=logging.INFO)

import rospy
import actionlib
from std_srvs.srv import Empty, EmptyResponse
from geometry_msgs.msg import PoseStamped
import rospkg
rospack = rospkg.RosPack()
ros_cosy_tracking_path = rospack.get_path('ros_cosy_tracking')

from ros_cosy_tracking.msg import StartCartesianImpedanceAction, StartCartesianImpedanceGoal


from ros_cosy_tracking.utils import posemsg_2_SE3, SE3_2_posiquat, posiquat_2_SE3, plot_se3_errors, plot_se3_abs
from ros_cosy_tracking.cam_extrinsics import T_ee_c

class PandaNode:

    def __init__(self, 
                 hostname: str, 
                 username: str, 
                 password: str, 
                 topic_kin: str, 
                 topic_object_pose: str,
                 T_co_ref: pin.SE3) -> None:
        
        print("hostname:", hostname)
        print("username:", username)
        print("password:", password)

        # panda_py objects
        self.desk = panda_py.Desk(hostname, username, password)
        self.connect_desk()

        # logic variables
        self.robot_controlled_flag = False
        self.last_state_time = 0.0
        self.kine_updated = False

        # panda_py controllers transforms
        self.T_b_ee_d = None
        self.T_co_ref:pin.SE3 = None
        self.T_ee_c = T_ee_c
        self.T_b_c = None

        # ROS
        self.kin_pub = rospy.Publisher(topic_kin, PoseStamped, queue_size=1)
        rospy.Timer(rospy.Duration(1.0/RATE_KINE), self.cb_timer_kine)
        rospy.Service('move_to_start', Empty, self.cb_move_to_base)
        rospy.Service('connect_desk', Empty, self.cb_connect_desk)
        rospy.Service('disconnect_desk', Empty, self.cb_disconnect_desk)
        # rospy.Subscriber(topic_object_tracks, Tracks, self.cb_tracks, queue_size=1)
        rospy.Subscriber(topic_object_pose, PoseStamped, self.cb_pose, queue_size=1)
        
        self._as = actionlib.SimpleActionServer(
            name='start_cartesian_impedance', 
            ActionSpec=StartCartesianImpedanceAction, 
            execute_cb=self.cb_start_cartesian_ctrl_action, 
            auto_start=False)
        self._as.start()

    def connect_desk(self):
        t = time.time()
        print('Connecting to desk...')
        self.desk.take_control()
        self.desk.unlock()
        self.desk.activate_fci()
        self.panda = panda_py.Panda(hostname)
        self.gripper = libfranka.Gripper(hostname)
        print(f'Connected! ({time.time() - t} s)')

    def disconnect_desk(self):
        print('Disconnecting to desk...')
        self.panda = None
        self.gripper = None
        self.desk.deactivate_fci()
        self.desk.lock()
        self.desk.release_control()
        print('Disconnected')

    def get_state_safe(self):
        # read current joint values
        # - if running controller, internal panda state is updated, just retrieve it
        # - otherwise, update it by reading on libfranka
        state = self.panda.get_state()
        if not self.robot_controlled_flag and self.last_state_time >= state.time.to_sec():
            state = self.panda.get_robot().read_once()
        self.last_state_time = state.time.to_sec()

        return state

    def cb_start_cartesian_ctrl_action(self, msg: StartCartesianImpedanceGoal):
        PRINT_EVERY = 200
        print('HOOOOOOOOOOOOOOOOOOOO')
        print('HOOOOOOOOOOOOOOOOOOOO')
        print('HOOOOOOOOOOOOOOOOOOOO')
        print('HOOOOOOOOOOOOOOOOOOOO')

        max_rtime = msg.max_runtime
        freq = msg.frequency
        
        if max_rtime == 0:
            # TODO: does not seem reasonable, how do we stop a context
            print('Attempting to create a context with infinit runtime -> abort')
            self._as.set_aborted()
            return

        self.robot_controlled_flag = True
        impedance = np.eye(6)
        impedance[range(0,3),range(0,3)] = msg.Kpt
        impedance[range(3,6),range(3,6)] = msg.Kpo
        ctrl = controllers.CartesianImpedance(
            filter_coeff=1.0,
            impedance=impedance
            )
        self.panda.start_controller(ctrl)

        # Set desired pose to the current one to start
        x_d = self.panda.get_position()
        q_d = self.panda.get_orientation()  # default representation is scalar last: xyzw
        i = 0
        
        T_b_ee_d_lst = []
        T_b_EE_m_lst = []
        with self.panda.create_context(frequency=freq, max_runtime=max_rtime) as ctx:
            while ctx.ok():
                x = self.panda.get_position()
                q = self.panda.get_orientation()  # default representation is scalar last: xyzw
                T_b_EE_m_lst.append(posiquat_2_SE3(x,q))
                if self.T_b_ee_d is not None:
                    x_d, q_d = SE3_2_posiquat(self.T_b_ee_d)
                    T_b_ee_d_lst.append(self.T_b_ee_d)
                else:
                    T_b_ee_d_lst.append(pin.SE3.Identity())
                    continue
                
                # Use default nullspace configuration
                ctrl.set_control(x_d, q_d)
                i += 1
                if i % PRINT_EVERY == 0:
                    print(f'{i/freq}/{max_rtime}')
                    print('x_d:', x_d)
                    print('q_d:', q_d)
        
        print('T_b_ee_d_lst[-1]')
        print(T_b_ee_d_lst[-1])
        print('T_b_EE_m_lst[-1]')
        print(T_b_EE_m_lst[-1])
        t_arr = np.arange(len(T_b_EE_m_lst)) / freq
        error_path = os.path.join(ros_cosy_tracking_path, 'out', 'errors_se3.png')
        abs_path = os.path.join(ros_cosy_tracking_path, 'out', 'abs_se3.png')
        plot_se3_errors(t_arr, T_b_ee_d_lst, T_b_EE_m_lst, error_path)
        plot_se3_abs(t_arr, T_b_ee_d_lst, T_b_EE_m_lst, abs_path)

        print('Stopping controller...')
        self.panda.stop_controller()
        print('Stopped')
        self.robot_controlled_flag = False

        self._as.set_succeeded()

    def cb_pose(self, msg: PoseStamped):
        # print('cb_pose')
        if not self.kine_updated:
            return
        # print('   cb_pose OK')

        T_bo = posemsg_2_SE3(msg.pose)

        if self.T_co_ref is None:
            T_co = (self.T_b_c.inverse() * T_bo).homogeneous
            self.T_co_ref:pin.SE3 = pin.SE3(T_co)
        T_oc_ref = self.T_co_ref.inverse()
        T_c_ee = self.T_ee_c.inverse()

        self.T_b_ee_d = T_bo * T_oc_ref * T_c_ee

    def cb_timer_kine(self, event=None):
        if self.panda is None:
            return
        
        t = rospy.Time.now()

        state = self.get_state_safe()
        self.T_b_ee_m = pin.SE3(np.array(state.O_T_EE).reshape((4,4)).T)
        self.T_b_c = self.T_b_ee_m * self.T_ee_c
        # print(f"self.T_b_c:{self.T_b_c}")

        pose_t = PoseStamped()
        pose_vec = pin.SE3ToXYZQUAT(self.T_b_c)
        pose_t.pose.position.x = pose_vec[0]
        pose_t.pose.position.y = pose_vec[1]
        pose_t.pose.position.z = pose_vec[2]
        pose_t.pose.orientation.x = pose_vec[3]
        pose_t.pose.orientation.y = pose_vec[4]
        pose_t.pose.orientation.z = pose_vec[5]
        pose_t.pose.orientation.w = pose_vec[6]
        # pose_t.header.seq = ?
        pose_t.header.stamp = t
        # pose_t.header.frame_id = ?

        self.kin_pub.publish(pose_t)

        self.kine_updated = True
        
    def cb_move_to_base(self, srv: Empty):
        self.robot_controlled_flag = True
        self.panda.move_to_start()
        self.robot_controlled_flag = False
        return EmptyResponse()
    
    def cb_connect_desk(self, msg: Empty):
        self.connect_desk()
        return EmptyResponse()
        
    def cb_disconnect_desk(self, msg: Empty):
        self.disconnect_desk()
        return EmptyResponse()


if __name__ == '__main__':
    rospy.init_node('panda_ctrl')
    topic_kin = 'pose_camera'
    topic_object_pose = 'target_object_pose'

    RATE_KINE = 500

    # Panda hostname/IP and Desk login information of your robot
    hostname = os.environ["PANDA_HOSTNAME"]
    username = os.environ["PANDA_USERNAME"]
    password = os.environ["PANDA_PASSWORD"]

    # TODO: read from yaml or orchestrator topic
    T_co_ref = pin.SE3.Identity()
    T_co_ref.translation[2] = 0.4
    # pose_co_ref = [
    #     -0.016252201050519943, 0.009602378122508526, 0.4408486783504486,
    #     0.4430050022551285, 0.5044145096723475, -0.5076611859052633, 0.5399931979578819
    # ]
    # T_co_ref = pin.XYZQUATToSE3(pose_co_ref)

    pn = PandaNode(hostname, username, password, topic_kin, topic_object_pose, T_co_ref)

    try:
        rospy.spin()
    except (rospy.ROSInterruptException, rospy.ROSInitException):
        pn.disconnect_desk()

