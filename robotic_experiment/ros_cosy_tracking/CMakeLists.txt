cmake_minimum_required(VERSION 3.0.2)
project(ros_cosy_tracking)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  genmsg 
  actionlib_msgs 
  actionlib
  std_msgs
  geometry_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################
# Generate messages in the 'msg' folder
add_message_files(
  FILES
  Track.msg
  Tracks.msg
  CosyDetection.msg
  CosyDetections.msg
  BoundingBox.msg
)

add_action_files(DIRECTORY action FILES StartCartesianImpedance.action)

###########
## Build ##
###########


include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_python_setup()

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
)

catkin_package(CATKIN_DEPENDS
  message_runtime
  std_msgs
  sensor_msgs
  geometry_msgs
)

# catkin_install_python(PROGRAMS nodes/panda_ctrl.py
#                       DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})