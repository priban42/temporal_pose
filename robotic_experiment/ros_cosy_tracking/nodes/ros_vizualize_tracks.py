#!/usr/bin/env python

import os
import threading
from pathlib import Path
from collections import defaultdict
import numpy as np
import pinocchio as pin

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image

from ros_cosy_tracking.msg import Tracks

#  roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 serial_no:=105322250337
#  rosrun image_view image_view image:=/camera/color/image_raw

YCBV_OBJECT_NAMES = {"obj_000001": "01_master_chef_can",
    "obj_000002": "02_cracker_box",
    "obj_000003": "03_sugar_box",
    "obj_000004": "04_tomatoe_soup_can",
    "obj_000005": "05_mustard_bottle",
    "obj_000006": "06_tuna_fish_can",
    "obj_000007": "07_pudding_box",
    "obj_000008": "08_gelatin_box",
    "obj_000009": "09_potted_meat_can",
    "obj_000010": "10_banana",
    "obj_000011": "11_pitcher_base",
    "obj_000012": "12_bleach_cleanser",
    "obj_000013": "13_bowl",
    "obj_000014": "14_mug",
    "obj_000015": "15_power_drill",
    "obj_000016": "16_wood_block",
    "obj_000017": "17_scissors",
    "obj_000018": "18_large_marker",
    "obj_000019": "19_large_clamp",
    "obj_000020": "20_extra_large_clamp",
    "obj_000021": "21_foam_brick"}
YCBV_OBJECT_NAMES_INV = {v: k for k, v in YCBV_OBJECT_NAMES.items()}

HOPE_OBJECT_NAMES = {"obj_000001": "AlphabetSoup",
    "obj_000002": "BBQSauce",
    "obj_000003": "Butter",
    "obj_000004": "Cherries",
    "obj_000005": "ChocolatePudding",
    "obj_000006": "Cookies",
    "obj_000007": "Corn",
    "obj_000008": "CreamCheese",
    "obj_000009": "GranolaBars",
    "obj_000010": "GreenBeans",
    "obj_000011": "Ketchup",
    "obj_000012": "MacaroniAndCheese",
    "obj_000013": "Mayo",
    "obj_000014": "Milk",
    "obj_000015": "Mushrooms",
    "obj_000016": "Mustard",
    "obj_000017": "OrangeJuice",
    "obj_000018": "Parmesan",
    "obj_000019": "Peaches",
    "obj_000020": "PeasAndCarrots",
    "obj_000021": "Pineapple",
    "obj_000022": "Popcorn",
    "obj_000023": "Raisins",
    "obj_000024": "SaladDressing",
    "obj_000025": "Spaghetti",
    "obj_000026": "TomatoSauce",
    "obj_000027": "Tuna",
    "obj_000028": "Yogurt"}
HOPE_OBJECT_NAMES_INV = {v: k for k, v in HOPE_OBJECT_NAMES.items()}


def ros_pose_to_SE3(pose):
    xyz_quat = np.array([pose.position.x,
    pose.position.y,
    pose.position.z,
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z,
    pose.orientation.w])
    T_co = pin.XYZQUATToSE3(xyz_quat).homogeneous
    return T_co

def SE3_to_ros_pose(T):
    T_pin = pin.SE3(T)
    xyz_qxyzw = pin.SE3ToXYZQUAT(T_pin)
    pose = Pose()
    pose.position.x = xyz_qxyzw[0]
    pose.position.y = xyz_qxyzw[1]
    pose.position.z = xyz_qxyzw[2]
    pose.orientation.x = xyz_qxyzw[3]
    pose.orientation.y = xyz_qxyzw[4]
    pose.orientation.z = xyz_qxyzw[5]
    pose.orientation.w = xyz_qxyzw[6]
    return pose

def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    if os.path.exists(example_dir / "meshes"):
        object_dirs = (example_dir / "meshes").iterdir()
    else:
        object_dirs = (example_dir.parent / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=HOPE_OBJECT_NAMES[label], mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def make_object_dataset_new(example_dir: Path, dataset='hope') -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    objects_dir = (example_dir).iterdir()
    for object_path in objects_dir:
        if object_path.suffix in {".obj", ".ply"}:
            obj_name = object_path.name[:-4]
            if dataset == 'hope':
                # label = obj_name
                label = HOPE_OBJECT_NAMES[obj_name]
            elif dataset == 'ycbv':
                label = YCBV_OBJECT_NAMES[obj_name]
            else:
                raise Exception(f"unknown dataset '{dataset}', required from ('hope', 'ycbv')")
            rigid_objects.append(RigidObject(label=label, mesh_path=object_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


class CosyViz:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.dataset = 'hope'
        MODELS_PATH = Path("/home/imitlearn/sandbox_mfourmy/local_data_happypose/bop_datasets/hopeVideo/models")
        self.poses = {}
        self.last_state = None
        self.last_time_stamp = None
        self.state_lock = threading.Lock()
        self.resolution = (480, 640)
        self.object_dataset = make_object_dataset_new(MODELS_PATH)
        self.K = np.array([[614.19298, 0.0, 326.26801],
                           [0.0, 614.19298, 238.851],
                           [0.0, 0.0, 1.0]])
        self.renderer = Panda3dSceneRenderer(self.object_dataset)
        self.last_dets = None
        self.last_rendered = True
        rospy.Subscriber("/tracks", Tracks, callback=self.cosy_dets_cb, queue_size=1)


    def rendering(self, predictions):
        camera_data = CameraData
        camera_data.K = self.K
        camera_data.TWC = Transform(np.eye(4))
        camera_data.resolution = self.resolution
        object_datas = []
        for label in predictions:
            if label != "Camera":
                for obj_inst in predictions[label]:
                    if isinstance(obj_inst, dict):
                        if obj_inst["valid"] == False:
                            continue
                        T_co = obj_inst["T_co"]
                    else:
                        T_co = obj_inst

                    object_datas.append(ObjectData(label=label, TWO=Transform(T_co)))
                    # object_datas.append(ObjectData(label='01_master_chef_can', TWO=TWO))
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

    def tracks_to_lists(self, tracks):
        T_cos = defaultdict(list)
        for track in tracks:
            T_co = ros_pose_to_SE3(track.pose)
            # Q = np.array(det.posecov.covariance).reshape((6, 6))
            if self.dataset == 'hope':
                obj_label = track.label
            elif self.dataset == 'ycbv':
                obj_label = track.label
            else:
                raise Exception(f"unknown dataset '{self.dataset}', required from ('hope', 'ycbv')")
            T_cos[obj_label].append(T_co)
        return T_cos

    def cosy_dets_cb(self, msg):
        with self.state_lock:
            if self.last_rendered:
                self.last_dets = msg
                self.last_rendered = False
    def render_dets(self):
        with self.state_lock:
            if self.last_dets == None or self.last_rendered:
                return None
            # rospy.loginfo(f'cosy_detection_cb called')
            all_T_cos = self.tracks_to_lists(self.last_dets.tracks)
            # all_T_cos = {"Cookies":[np.array(((1, 0, 0, 0),
            #                                       (0, 0, 1, 0),
            #                                       (0, 1, 0, 0.5),
            #                                       (0, 0, 0, 1)))]}
            renderings = self.rendering(all_T_cos)
            # cv2.imshow('img', renderings.rgb)
            # cv2.waitKey(0)
            # img_msg = self.cv_bridge.cv2_to_imgmsg(renderings.rgb, encoding="bgr8")
            rospy.loginfo(f'renderings.rgb.shape:{renderings.rgb.shape}, len(all_T_cos):{len(all_T_cos)}, dtype:{renderings.rgb.dtype}')
            rospy.loginfo(f'np.min(renderings.rgb):{np.min(renderings.rgb)}, np.max(renderings.rgb):{np.max(renderings.rgb)}')
            # img_msg = self.cv_bridge.cv2_to_imgmsg(np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8),  encoding="bgr8")
            # self.image_pub.publish(img_msg)
            self.last_rendered = True
            return renderings.rgb

def main():
    RATE = 15 # Hz
    k = 0
    cosyViz = CosyViz()
    rospy.init_node("tracksviz")
    image_pub = rospy.Publisher('tracksviz', Image, queue_size=1)
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown():
        rgb = cosyViz.render_dets()
        if rgb is not None:
            image_pub.publish(cosyViz.cv_bridge.cv2_to_imgmsg(rgb, encoding="rgb8"))
        k += 1
        rate.sleep()

if __name__ == "__main__":
    main()
