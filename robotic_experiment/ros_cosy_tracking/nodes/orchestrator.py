#!/usr/bin/env python

import rospy
import actionlib
from ros_cosy_tracking.msg import StartCartesianImpedanceAction, StartCartesianImpedanceGoal


if __name__ == '__main__':
    rospy.init_node('orchestrator')
    client = actionlib.SimpleActionClient('start_cartesian_impedance', StartCartesianImpedanceAction)
    print('orchestrator WAITING')
    client.wait_for_server()
    print('orchestrator OK!!!')

    goal = StartCartesianImpedanceGoal()
    # Fill in the goal here
    goal.max_runtime = 30.0
    goal.frequency = 500
    goal.Kpt = 1.5*100
    goal.Kpo = 1.5*20
    client.send_goal(goal)
    # client.wait_for_result(rospy.Duration.from_sec(5.0))