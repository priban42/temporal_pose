# ROS Cosy Tracking

This folder contains the ROS implementation used during the robotic experiments for the qualitative evaluation of the method proposed in the thesis. The implementation is designed for ROS Noetic.

For useful commands, refer to `notes.txt`.

## Nodes

- **ros_cosycov.py**: Utilizes CosyPose to estimate object poses and publishes the results.
- **ros_gtsam.py**: Applies our method to create temporally consistent object tracks using messages from `ros_cosycov.py`.
- **ros_vizualize_cosy.py**: Publishes image renders of the CosyPose detections.
- **ros_vizualize_tracks.py**: Publishes image renders of the refined object tracks.
- **panda_ctrl.py**: Facilitates the control of the Panda Franka Emika robot.