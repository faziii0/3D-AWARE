import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# === Correct Paths to JSON files ===
FRONT_POSE_PATH = '/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/002/camera/front_camera/poses.json'
LIDAR_POSE_PATH = '/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/002/lidar/poses.json'

def quaternion_to_matrix(quat):
    r = R.from_quat([quat['x'], quat['y'], quat['z'], quat['w']])
    return r.as_matrix()

def build_transform(position, heading):
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(heading)
    T[:3, 3] = [position['x'], position['y'], position['z']]
    return T

def get_tr_velo_to_cam_all():
    with open(FRONT_POSE_PATH, 'r') as f:
        front_poses = json.load(f)

    with open(LIDAR_POSE_PATH, 'r') as f:
        lidar_poses = json.load(f)

    tr_velo_to_cam_all = []

    for i in range(len(front_poses)):
        T_front = build_transform(front_poses[i]['position'], front_poses[i]['heading'])
        T_lidar = build_transform(lidar_poses[i]['position'], lidar_poses[i]['heading'])

        T_lidar_inv = np.linalg.inv(T_lidar)
        T_velo_to_cam = T_front @ T_lidar_inv

        tr_velo_to_cam_all.append(T_velo_to_cam)

    return tr_velo_to_cam_all

