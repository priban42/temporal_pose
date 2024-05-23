import bpy
import os
import pickle
import csv

import mathutils
from mathutils import Matrix, Vector
import time
import csv
import json
import shutil
import numpy as np
from CameraIntrinsics import get_intrinsics_as_K_matrix
from Undistorter import get_depth_undistorter
import cv2
from pathlib import Path


def __export_json(path, dict, indent=2):
    json_object = json.dumps(dict, indent=indent)
    with open(path, "w") as outfile:
        outfile.write(json_object)

def __export_json_line_per_entry(path, dict, indent=2):
    with open(path, "w") as outfile:
        outfile.write('{\n')

        for i, key in enumerate(dict.keys()):
            if i != 0:
                outfile.write(",\n")
            entry = json.dumps(dict[key])
            #line = " "*indent + key + ": " + entry + '\n'
            line = f'{" "*indent}"{key}": {entry}'
            outfile.write(line)
        outfile.write('\n}')
    with open(path, "r") as test_json:
        data = json.load(test_json)


def export_camera_intrinsics(dataset_path:Path):
    DECIMALS = 6
    K = get_intrinsics_as_K_matrix()
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    #depth_scale = bpy.context.scene.world.mist_settings.depth*1000/(2**16)
    depth_scale = round(bpy.context.scene.world.mist_settings.depth * 1000 / (2 ** 16), 2)
    #depth_scale = bpy.data.scenes["Scene"].node_tree.nodes["Math"].inputs[1].default_value
    dictionary = {"cx": round(cx, DECIMALS),
                  "cy": round(cy, DECIMALS),
                  "depth_scale": 0.1,
                  "fx": round(fx, DECIMALS),
                  "fy": round(fy, DECIMALS),
                  "height": height,
                  "width": width
                  }
    __export_json(dataset_path / "camera.json", dictionary)


def __save_scene_camera(scene, base_path:Path, depth_scale):
    DECIMALS = 5
    dictionary = {}
    K = get_intrinsics_as_K_matrix()
    cam_K = [round(item, DECIMALS) for row in K for item in row]
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_current = frame
        #bpy.context.view_layer.update()
        bpy.ops.wm.tool_set_by_id(name="builtin.rotate")
        T_c2w_blender = __get_current_frame_T_c2w()
        T_c2w_cv2 = __blender_T_to_cv2(T_c2w_blender)
        T_w2c =T_c2w_cv2.inverted()
        R = __T_to_R(T_w2c)
        t = __T_to_t(T_w2c)

        cam_R_w2c = R.flatten()
        cam_t_w2c = t.flatten()*1000 #  converting from meters to milimeters

        entry = {"cam_K": cam_K, "cam_R_w2c": cam_R_w2c.tolist(), "cam_t_w2c": cam_t_w2c.tolist(), "depth_scale": depth_scale}
        dictionary[str(frame)] = entry
    __export_json_line_per_entry(base_path / "scene_camera.json", dictionary)

def __save_scene_gt(scene, base_path:Path, arrangement):
    DECIMALS = 5
    dictionary = {}
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_current = frame
        #bpy.context.view_layer.update()
        bpy.ops.wm.tool_set_by_id(name="builtin.rotate")
        entry = []
        for object in arrangement.all_objects:
            if not object.hide_render:
                obj_id = int(object.name_full.split(".")[0][4:])
                T_m2w = object.matrix_world
                T_w2m = T_m2w.inverted()  # scale fix
                T_c2w_blender = __get_current_frame_T_c2w()
                T_c2w_cv2 = __blender_T_to_cv2(T_c2w_blender)
                T_c2m_cv2 = T_w2m @ T_c2w_cv2
                T_m2c_cv2 = T_c2m_cv2.inverted()
                R = __T_to_R(T_m2c_cv2)
                t = __T_to_t(T_m2c_cv2)
                cam_R_m2c = R.flatten()
                cam_t_m2c = t.flatten()*1000  # converting from meters to milimeters
                entry.append({"cam_R_m2c": cam_R_m2c.tolist(), "cam_t_m2c": cam_t_m2c.tolist(), "obj_id": obj_id})
        dictionary[str(frame)] = entry
    __export_json_line_per_entry(base_path / "scene_gt.json", dictionary)

def __get_current_frame_T_c2w():
    camera = bpy.context.scene.camera
    T_w2c = camera.matrix_world
    return T_w2c

def __T_to_R(T):
    R = np.array(T)[0:3, 0:3]
    return R
def __T_to_t(T):
    t = np.array(T)[0:3, 3]
    return t

def __blender_T_to_cv2(T):
    T_cv2blender = mathutils.Matrix(((1, 0, 0, 0),
                              (0, -1, 0, 0),
                              (0, 0, -1, 0),
                              (0, 0, 0, 1)))
    return T@T_cv2blender

def test():
    T_c2w = __get_current_frame_T_c2w()
    print(T_c2w)
    print(__T_to_R(T_c2w))
    print(__T_to_t(T_c2w))


def __render_speed_settings(scene, frame_end, speed=1):
    scene.render.frame_map_new = int(100 / speed)
    scene.frame_start = int(scene.frame_start / speed)
    scene.frame_end = int(frame_end / speed)

def __refresh_dir(path:Path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)

def __soft_refresh_dir(path:Path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        return
    os.makedirs(path)

def rgb_render_settings():
    bpy.context.scene.camera.scale = Vector((1, 1, 1))
    bpy.context.scene.camera.data.type = 'PERSP'
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.view_settings.view_transform = 'Filmic'
    bpy.context.scene.view_settings.gamma = 1

def depth_render_settings():
    bpy.context.scene.camera.scale = Vector((1, 1, 1))
    bpy.context.scene.camera.data.type = 'PERSP'
    # bpy.context.scene.camera.data.type = 'ORTHO'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.display_settings.display_device = 'None'
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.gamma = 1
    bpy.context.scene.view_settings.use_curve_mapping = False

undistorter = get_depth_undistorter(get_intrinsics_as_K_matrix(),
                                    (bpy.context.scene.render.resolution_y,
                                     bpy.context.scene.render.resolution_x))

def undistort_depth_images(depth_path:Path, undistorter):

    for img_name in os.listdir(depth_path):
        img_path = depth_path / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = (undistorter*img).astype('uint16')
        cv2.imwrite(str(img_path), img)


def generate_scene_entry(dataset_path:Path, arrangement, scene_number, speed=1, subdir_name='test', use_depth = False):
    scene = bpy.data.scenes['Scene']
    # scene.node_tree.nodes["Switch"].check = False
    frame_end = scene.frame_end
    scene_path = dataset_path / subdir_name / f'{scene_number:06}'
    rgb_path = scene_path / 'rgb'
    depth_path = scene_path / 'depth'
    scene.node_tree.nodes["Switch"].check = False
    __refresh_dir(scene_path)
    __refresh_dir(rgb_path)
    if use_depth:
        __refresh_dir(depth_path)
    __render_speed_settings(scene, frame_end, speed=speed)
    depth_scale = round(bpy.context.scene.world.mist_settings.depth * 1000 / (2 ** 16), 2)
    __save_scene_camera(scene, scene_path, depth_scale)
    __save_scene_gt(scene, scene_path, arrangement)
    rgb_render_settings()
    bpy.context.scene.render.filepath = str(rgb_path / "######") #  #### for number padding
    bpy.ops.render.render(animation=True, use_viewport=True)

    if use_depth:
        depth_render_settings()
        bpy.context.scene.render.filepath = str(depth_path / "######")  # #### for number padding
        scene.node_tree.nodes["Switch"].check = True
        bpy.ops.render.render(animation=True, use_viewport=True)
        undistort_depth_images(depth_path, undistorter)

    scene.node_tree.nodes["Switch"].check = False
    rgb_render_settings()
