import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy as np
from numpy.typing import ArrayLike
import cv2 as cv
from pyrcareworld.attributes import CameraAttr

#Unity Space Rotation
T_UNITY = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


def world2camera(world_position : ArrayLike, camera : CameraAttr, intrinsics: ArrayLike) -> np.ndarray:
    P = np.zeros((3,4))
    P[:3,:3] = intrinsics

    T = np.array(camera.data["local_to_world_matrix"])
    T = np.linalg.inv(T)

    P = P @ T

    X = np.empty(4)
    X[:3] = world_position
    X[3] = 1

    x = P @ X
    x /= x[2]

    point = x[:2].astype(int)
    point[1] = 512 - point[1]

    return point

def get_color(camera:CameraAttr) -> np.ndarray:
    if "rgb_decoded" in camera.data:
        img = camera.data["rgb_decoded"]
    else:
        rgb = np.frombuffer(camera.data["rgb"], dtype=np.uint8)
        img = cv.imdecode(rgb, cv.IMREAD_COLOR)

        camera.data["rgb_decoded"] = img

    return img

def get_depth_exr(camera:CameraAttr) -> np.ndarray:
    if "depth_exr_decoded" in camera.data:
        depth = camera.data["depth_exr_decoded"]
    else:
        depth_bytes = camera.data["depth_exr"]

        temp_file_path = "img.exr"

        with open(temp_file_path, "wb") as f:
            f.write(depth_bytes)
        depth = cv.imread(temp_file_path, cv.IMREAD_UNCHANGED)

        os.remove(temp_file_path)

        camera.data["depth_exr_decoded"] = depth

    return depth

def get_positions(camera:CameraAttr, intrinsics_inv:ArrayLike, img_depth:ArrayLike, coordinates:ArrayLike) -> np.ndarray:
    coordinates = np.asarray(coordinates)
    intrinsics_inv = np.asarray(intrinsics_inv)
    img_depth = np.asarray(img_depth)

    T = np.array(camera.data["local_to_world_matrix"])
    T = T@T_UNITY

    original_shape = np.array(coordinates.shape)
    if coordinates.ndim != 2:
        coordinates = coordinates.reshape(-1, 2)
    
    positions = np.empty((coordinates.shape[0],3), np.float32)
    for i, coord in enumerate(coordinates):
        coord = np.array(coord)
        
        point_int = np.around(coord).astype(int)
        Z = img_depth[point_int[1], point_int[0]]

        x = np.empty(3)
        x[0] = coord[0]
        x[1] = coord[1]
        x[2] = 1
        x *= Z

        X = intrinsics_inv@x

        X_hom = np.empty(4)
        X_hom[:3] = X
        X_hom[3] = 1
        
        X = (T@X_hom)[:3]

        positions[i] = X

    shape = np.append(original_shape[:-1], [3])
    positions = positions.reshape(shape)

    return positions