#!/usr/bin/env python

import time
import numpy as np
import torch
import pinocchio as pin

from happypose.toolbox.inference.types import ObservationTensor
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import CosyPoseWrapper

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovariance

from ros_cosy_tracking.msg import CosyDetection, CosyDetections


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CosyPoseCovNode:

    def __init__(self, topic_cosy, topic_color, topic_color_info) -> None:
        rospy.init_node('ros_cosycov', anonymous=True)

        ds_name = 'ycbv'
        renderer_type, n_workers = 'panda3d', 1
        # renderer_type, n_workers = 'bullet', 0  # Not working
        self.cosypose = CosyPoseWrapper(
            ds_name,
            renderer_type=renderer_type,
            n_workers=n_workers
        )
        self.cv_bridge = CvBridge()

        self.pub_dets = rospy.Publisher(topic_cosy, CosyDetections, queue_size=1)

        color_sub = message_filters.Subscriber(topic_color, Image)
        color_info_sub = message_filters.Subscriber(topic_color_info, CameraInfo)

        ts = message_filters.TimeSynchronizer([color_sub, color_info_sub], 2)
        ts.registerCallback(self.callback_image)

        self.last_time_stamp = None

    def callback_image(self, color_msg: Image, cam_info: CameraInfo):
        print(f'device: {device}')
        print(color_msg.header.stamp)
        if self.last_time_stamp is not None:
            dt_cb = (color_msg.header.stamp - self.last_time_stamp).to_sec()
            print('dt_cb (ms)', 1e3*dt_cb)
        self.last_time_stamp = color_msg.header.stamp
        rgb = self.cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')

        obs = ObservationTensor.from_numpy(
            rgb=rgb,
            K=np.array(cam_info.K).reshape((3, 3))
        )
        obs.to(device)
        
        detections = self.cosypose.pose_predictor.detector_model.get_detections(
            obs, 
            detection_th=0.9,
            mask_th=0.7,
            output_masks=True
        )

        # HACK: filter out the wooden object (mistake with table)
        labels_rm = ['ycbv-obj_000016']
        df = detections.infos
        df = df[~df.label.isin(labels_rm)]
        detections = detections[df.index.tolist()]

        cosy_dets = CosyDetections()
        cosy_dets.header = color_msg.header

        if len(detections.infos) == 0:
            self.pub_dets.publish(cosy_dets)
            return

        pixel_counts = detections.tensors['masks'].sum(axis=2).sum(axis=1)
        bboxes = detections.tensors['bboxes']

        data_TCO, extra_data = self.cosypose.pose_predictor.run_inference_pipeline(
            obs, detections=detections, run_detector=False,
            n_refiner_iterations=3
        )
        print('  DT run_inference_pipeline (ms)', extra_data["timing_str"])

        # Parsing cosy results into detection messages

        data_TCO = data_TCO.cpu()
        pixel_counts = pixel_counts.cpu()
        bboxes = bboxes.int().cpu()

        for i in range(len(data_TCO.infos)):
            # 4x4 tensor -> [tx,ty,tz, qx,qy,qz,qw] pose representations
            pose_vec = pin.SE3ToXYZQUAT(pin.SE3(data_TCO.poses[i].numpy()))

            det = CosyDetection()
            det.posecov = PoseWithCovariance()
            det.posecov.pose.position.x = pose_vec[0]
            det.posecov.pose.position.y = pose_vec[1]
            det.posecov.pose.position.z = pose_vec[2]
            det.posecov.pose.orientation.x = pose_vec[3]
            det.posecov.pose.orientation.y = pose_vec[4]
            det.posecov.pose.orientation.z = pose_vec[5]
            det.posecov.pose.orientation.w = pose_vec[6]

            det.obj_label = data_TCO.infos.loc[i, 'label']
            det.score = data_TCO.infos.loc[i, 'score']
            det.mask_pixel_count = pixel_counts[i].item()
            det.bbox.xmin = bboxes[i,0].item()
            det.bbox.ymin = bboxes[i,1].item()
            det.bbox.xmax = bboxes[i,2].item()
            det.bbox.ymax = bboxes[i,3].item()

            # TODO: covariance computation
            cov = np.eye(6)
            det.posecov.covariance = cov.flatten().tolist()

            cosy_dets.dets.append(det)

        self.pub_dets.publish(cosy_dets)



if __name__ == '__main__':

    camera_name = 'camera'
    # camera_name = 'camera/realsense2_camera'
    # camera_name = 'camera_scene'

    topic_cosy = 'cosy_detections'
    topic_color = f'/{camera_name}/color/image_raw'
    topic_color_info = f'/{camera_name}/color/camera_info'

    CosyPoseCovNode(topic_cosy, topic_color, topic_color_info)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

