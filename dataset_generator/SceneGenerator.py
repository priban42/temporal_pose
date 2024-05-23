import bpy
import os
import pickle
import csv
import mathutils
import numpy
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
import random
from mathutils.bvhtree import BVHTree
import pinocchio as pin

def worldBoundingBox(obj):
    """returns the corners of the bounding box of an object in world coordinates"""
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

def objectsOverlap(obj1, obj2):
    """returns True if the object's bounding boxes are overlapping"""
    vert1 = worldBoundingBox(obj1)
    vert2 = worldBoundingBox(obj2)
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7)]

    bvh1 = BVHTree.FromPolygons(vert1, faces)
    bvh2 = BVHTree.FromPolygons(vert2, faces)

    return bool(bvh1.overlap(bvh2))

def object_colides_with_collection(obj, collection):
    for c_obj in collection.objects:
        if c_obj != obj:
            if objectsOverlap(obj, c_obj):
                return True
    return False

def refresh_collection(name):
    collection = bpy.data.collections.get(name)
    if collection != None:
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)

    new_collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(new_collection)
    return new_collection

def copy_object_to_collection(object, collection):
    new_obj = object.copy()
    new_obj.data = object.data.copy()
    collection.objects.link(new_obj)
    return new_obj

def move_obj_randomly(obj, box_size = 0.3):
    x = random.uniform(-box_size / 2, box_size / 2)
    y = random.uniform(-box_size / 2, box_size / 2)
    z = obj.location[2]
    obj.location = Vector((x, y, z))
    obj.rotation_euler.z = random.uniform(-180, 180)
    bpy.context.view_layer.update()

def arrange_static_scene(models_collection, obj_count = 7, density = 1):
    output_collection = refresh_collection("Output")
    for i in range(obj_count):
        rnd_obj = models_collection.objects[random.randint(0, len(models_collection.objects) - 1)]
        new_obj = copy_object_to_collection(rnd_obj, output_collection)
        move_obj_randomly(new_obj, obj_count*0.04/density)
        for x in range(100):
            # print(f"{i}: {x}")
            move_obj_randomly(new_obj, obj_count*0.04/density)
            if not object_colides_with_collection(new_obj, output_collection):
                break
        if object_colides_with_collection(new_obj, output_collection):
            raise Exception(f"Could not find a place for object {new_obj.name_full} that is not in collision with {output_collection.name_full}.")

def get_point_on_hemi_sphere(r = 1, base_vector= np.array((1, 0, 0)), weight=1):
    min_height = 0.1

    vec = np.random.randn(3) + base_vector*weight
    vec /= np.linalg.norm(vec, axis=0)
    vec *= r
    vec[2] = abs(vec[2])
    if vec[2] <= min_height:
        vec[2] += min_height
    return vec

def look_at_centre(camera):
    T_wc = np.asarray(camera.matrix_world)
    w = np.linalg.norm(pin.log3(T_wc[:3, :3]))
    # print(f"T_wc: {T_wc}")
    direction = T_wc[:3, 3]/np.linalg.norm(T_wc[:3, 3])
    new_w = direction*w
    # print(f"w: {w}")
    R = pin.exp3(new_w)
    T_wc[:3, :3] = R
    camera.matrix_world = Matrix(T_wc)

def edit_matrix_world(mw, fw_vect, right_vect, up_vect):
    new_mw = np.array(mw)
    new_mw[0:3, 0] = np.array(right_vect)
    new_mw[0:3, 1] = np.array(up_vect)
    new_mw[0:3, 2] = np.array(fw_vect)
    return Matrix(new_mw)

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def rotate_camera_toward_point(camera):
    z1 = camera.rotation_euler.z
    T_wc = np.asarray(camera.matrix_world)
    point = Vector((0.0, 0, 0.0))
    fw_vect = -(point - camera.location)
    fw_vect.normalize()
    if fw_vect == Vector((0, 0, 1)):
        right_vect = Vector((0, 1, 0))
    else:
        right_vect = -(fw_vect.cross(Vector((0, 0, 1))))
    right_vect.normalize()
    up_vect = fw_vect.cross(right_vect)
    up_vect.normalize()

    # R = numpy.array((fw_vect, right_vect, up_vect))
    # # print(f"R:{R}")
    # # r,p,y = rot2eul(T_wc[:3, :3]@R.T)
    # # camera.rotation_euler.x += r
    # # camera.rotation_euler.y += p
    # # camera.rotation_euler.z += y

    camera.matrix_world = edit_matrix_world(camera.matrix_world, fw_vect, right_vect, up_vect)
    bpy.context.view_layer.update()
    z2 = camera.rotation_euler.z
    # print(f"{180*z1/np.pi}, {180*z2/np.pi}")
    if abs(z1 - z2) > np.pi:
        if z1 > 0:
            camera.rotation_euler.z += 2*np.pi
        else:
            camera.rotation_euler.z -= 2 * np.pi
    bpy.context.view_layer.update()


def generate_random_camera_trajectory(length, sphere_size = 0.5, weight=4, density=10, base_vector_offset = np.zeros((3))):
    camera = bpy.data.objects.get("Camera")
    camera.animation_data_clear()
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.frame_start = 1
    scene.frame_end = length + 1
    base_vector = np.array((1, 0, 0))
    for x in range(3 + length//density):
        frame = x * density
        base_vector = get_point_on_hemi_sphere(sphere_size, base_vector, weight=weight)
        camera.location = Vector(base_vector)
        bpy.context.view_layer.update()
        rotate_camera_toward_point(camera)
        camera.location = camera.location + Vector(base_vector_offset)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def animate_object(obj, trajectory, length):

    for i, entry in enumerate(trajectory):
        frame = i
        obj.matrix_world = entry["SE3"]
        obj.location = entry["SE3"][:3, 3]
        obj.location = obj.location*0.25
        obj.location.z += 0.55
        bpy.context.view_layer.update()
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        if i == length:
            # print(f"here{length}")
            break

def create_dynamic_scene(trajectories, t = 0, length = 350):

    collection = bpy.data.collections.get("Output")
    for i, obj in enumerate(collection.objects):
        trajectory = trajectories[t*10 + i]
        animate_object(obj, trajectory, length)
        generate_random_camera_trajectory(length, sphere_size = 1.0, weight=6, density=30, base_vector_offset=np.array((0, 0, 0.5)))