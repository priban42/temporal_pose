#!/usr/bin/env python

import threading
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d

from happypose.toolbox.datasets.datasets_cfg import make_object_dataset
from happypose.toolbox.inference.types import PoseEstimatesType

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped

from ros_cosy_tracking.msg import CosyDetections, Tracks
from ros_cosy_tracking.utils import posemsg_2_SE3

cosy_enabled = True
tracks_enabled = True

class CosyViz:
    def __init__(self, 
                 topic_cosy: str, 
                 topic_tracks: str,
                 topic_color: str, 
                 topic_color_info: str,
                 topic_camera_pose: str,
                 topic_viz_cosy: str,
                 topic_viz_tracks: str):
        rospy.init_node("cosyviz")

        self.cv_bridge = CvBridge()
        # MODELS_PATH = Path("/home/imitlearn/sandbox_mfourmy/local_data_happypose/bop_datasets/hopeVideo/models")
        self.last_state = None
        self.last_time_stamp = None
        self.state_lock = threading.Lock()
        self.track_lock = threading.Lock()
        self.object_dataset = make_object_dataset('ycbv')
        all_obj_labels = set(self.object_dataset.label_to_objects.keys())
        self.renderer = Panda3dSceneRenderer(self.object_dataset, preload_labels=all_obj_labels)
        self.is_rendering = False

        if cosy_enabled:
            rospy.Subscriber(topic_cosy, CosyDetections, callback=self.cb_cosy_dets, queue_size=1)
        if tracks_enabled:
            tracks_sub = message_filters.Subscriber(topic_tracks, Tracks)
            camera_pose_sub = message_filters.Subscriber(topic_camera_pose, PoseStamped)
            approx_ts = message_filters.ApproximateTimeSynchronizer([tracks_sub, camera_pose_sub], 50, 0.1, allow_headerless=False)
            approx_ts.registerCallback(self.cb_tracks)

        color_sub = message_filters.Subscriber(topic_color, Image)
        color_info_sub = message_filters.Subscriber(topic_color_info, CameraInfo)
        ts = message_filters.TimeSynchronizer([color_sub, color_info_sub], 2)
        ts.registerCallback(self.cb_camera)

        self.cosy_viz_pub = rospy.Publisher(topic_viz_cosy, Image, queue_size=1)
        self.tracks_viz_pub = rospy.Publisher(topic_viz_tracks, Image, queue_size=1)

        self.last_dets = None
        self.last_tracks = None
        self.last_K = None

        self.last_cosy_render = None
        self.last_tracks_render = None
        self.new_detection = False
        self.new_track = False

        
    def rendering(self, predictions, K: np.ndarray, resolution: tuple):
        camera_data = CameraData(
            K=K,
            TWC=Transform(np.eye(4)),
            resolution=resolution,
        )
        object_datas = []
        for label in predictions:
            for obj_inst in predictions[label]:
                T_co = obj_inst["T_co"]
                object_datas.append(ObjectData(label=label, TWO=Transform(T_co)))

        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((0.6, 0.6, 0.6, 1)),
            ),
        ]
        renderings = self.renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]
        return renderings

    def dets_to_lists(self, dets):
        T_cos = defaultdict(list)
        px_counts = defaultdict(list)
        for det in dets:
            T_co = posemsg_2_SE3(det.posecov.pose).homogeneous
            obj_label = det.obj_label
            pixel_count = det.mask_pixel_count
            T_cos[obj_label].append({"T_co":T_co})
            px_counts[obj_label].append(pixel_count)
        return T_cos, px_counts
    
    def tracks_to_list(self, tracks, pose_bc: Pose):
        T_bc = posemsg_2_SE3(pose_bc)
        T_cos = defaultdict(list)
        for track in tracks.tracks:
            T_bo = posemsg_2_SE3(track.pose)
            obj_label = track.label
            T_co = T_bc.inverse() * T_bo
            T_cos[obj_label].append({"T_co":T_co, "id":track.track_id})
        return T_cos

    def cb_cosy_dets(self, msg):
        self.new_detection = True
        with self.state_lock:
            self.last_dets = msg
    
    def cb_tracks(self, track_msg, camera_pose_msg):
        self.new_track = True
        with self.track_lock:
            self.last_tracks = self.tracks_to_list(track_msg, camera_pose_msg.pose)

    def overlay_render(self, rgb, render):
        alpha = 0.5
        rgb = alpha*rgb + (1-alpha)*render
        rgb = rgb.astype(np.uint8)
        return rgb

    def cb_camera(self, color_msg: Image, cam_info: CameraInfo):
        rgb = self.cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        self.last_resolution = (rgb.shape[0], rgb.shape[1])
        self.last_K = np.array(cam_info.K).reshape((3, 3))
            
        # alpha blending
        if cosy_enabled:
            if self.last_cosy_render is not None:
                img_cosy = self.overlay_render(rgb, self.last_cosy_render)
                img_msg = self.cv_bridge.cv2_to_imgmsg(img_cosy, encoding="rgb8")
                img_msg.header = color_msg.header
                self.cosy_viz_pub.publish(img_msg)
        if tracks_enabled:
            if self.last_tracks_render is not None:
                img_gtsam = self.overlay_render(rgb, self.last_tracks_render)
                self.draw_track_ids(img_gtsam, self.last_tracks, self.last_K)
                img_msg = self.cv_bridge.cv2_to_imgmsg(img_gtsam, encoding="rgb8")
                img_msg.header = color_msg.header
                self.tracks_viz_pub.publish(img_msg)
   
    def render_dets(self):
        # with self.state_lock:
        if self.new_detection:
            if self.last_dets is None or self.last_K is None:
                return None
            # rospy.loginfo(f'cosy_detection_cb called')
            all_T_cos, all_px_counts = self.dets_to_lists(self.last_dets.dets)

            renderings = self.rendering(all_T_cos, self.last_K, self.last_resolution)
            self.last_cosy_render = renderings.rgb
            self.new_detection = False

    def draw_track_ids(self, img, predictions, K):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for obj_label in predictions:
            for obj_idx in range(len(predictions[obj_label])):
                uvw = K @ predictions[obj_label][obj_idx]["T_co"].translation
                uv = tuple((uvw[:2] / uvw[2]).astype(int))
                obj_id = predictions[obj_label][obj_idx]['id']
                cv2.putText(img, str(obj_id), org=tuple(uv), fontFace=font, fontScale=2, color=(0, 255, 0), thickness=4, lineType=2)

    def render_tracks(self):
        if self.new_track:
            if self.last_tracks is None or self.last_K is None:
                return None
            renderings = self.rendering(self.last_tracks, self.last_K, self.last_resolution)
            self.last_tracks_render = renderings.rgb
            self.new_track = False


def main():
    topic_cosy = 'cosy_detections'
    camera_name = 'camera'
    topic_color = f'/{camera_name}/color/image_raw'
    topic_color_info = f'/{camera_name}/color/camera_info'
    topic_camera_pose = f'/pose_camera'
    topic_tracks = 'tracks'

    cosy_viz = 'cosyviz'
    tracks_viz = 'tracksviz'
    
    RATE = 20 # Hz
    k = 0
    node = CosyViz(topic_cosy, topic_tracks, topic_color, topic_color_info, topic_camera_pose, cosy_viz, tracks_viz)
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown():
        if node.new_detection and cosy_enabled:
            rgb = node.render_dets()
        if node.new_track and tracks_enabled:
            rgb = node.render_tracks()
        k += 1
        rate.sleep()

if __name__ == "__main__":
    main()
