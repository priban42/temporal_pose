#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-02-19
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
from movement_primitives.dmp import CartesianDMP
import pinocchio as pin
import pickle
# import matplotlib.pyplot as plt


def sample_trajectory(number_of_waypoints=10, poses_per_waypoint=10):
    """Sample a trajectory with random waypoints and random DMPs.
    Return array of vectors representing pose with XYZQuat."""
    waypoints = [pin.SE3.Random() for _ in range(number_of_waypoints)]
    waypoints_velocities = [np.random.rand(6) - 0.5 for _ in range(number_of_waypoints)]
    waypoints_accelerations = [
        np.random.rand(6) - 0.5 for _ in range(number_of_waypoints)
    ]
    poses = []
    dmp = CartesianDMP(dt=1 / (poses_per_waypoint * 10))
    dmp.set_weights(1000 * (np.random.rand(*dmp.get_weights().shape) - 0.5))
    for i in range(1, number_of_waypoints):

        dmp.configure(
            start_y=pin.SE3ToXYZQUAT(waypoints[i - 1]),
            start_yd=waypoints_velocities[i - 1],
            start_ydd=waypoints_accelerations[i - 1],
            goal_y=pin.SE3ToXYZQUAT(waypoints[i]),
            goal_yd=waypoints_velocities[i],
            goal_ydd=waypoints_accelerations[i],
        )
        new_poses = dmp.open_loop(run_t=1.0)[1].tolist()[::10]
        for pose in new_poses:
            entry = {"XYZQUATT":pose, "SE3":pin.XYZQUATToSE3(pose).homogeneous}
            poses.append(entry)
    return np.asarray(poses)

def main():
    trajectories = []
    for x in range(100):
        trajectory = sample_trajectory(3, 200)
        trajectories.append(trajectory)
        print(f"\r{x} done", end='')
    with open('trajectories.p', 'wb') as file:
        pickle.dump(trajectories, file)

if __name__ == "__main__":
    main()



