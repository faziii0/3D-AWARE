from pandaset.dataset import DataSet
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import json
from os.path import *
import os
from tqdm import tqdm
import gc
import pickle


def compose(T, R, Z):
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError('Expecting shape (%d,%d) for rotations' % (n, n))
    A = np.eye(n + 1)
    ZS = np.diag(Z)
    A[:n, :n] = np.dot(R, ZS)
    A[:n, n] = T[:]
    return A


def quat2mat(q):
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X; wY = w * Y; wZ = w * Z
    xX = x * X; xY = x * Y; xZ = x * Z
    yY = y * Y; yZ = y * Z; zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def _heading_position_to_mat(heading, position):
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    transform_matrix = compose(np.array(pos), quat2mat(quat), [1.0, 1.0, 1.0])
    return transform_matrix


def lidar_points_to_ego(points, lidar_pose):
    lidar_pose_mat = _heading_position_to_mat(
        lidar_pose['heading'], lidar_pose['position'])
    transform_matrix = np.linalg.inv(lidar_pose_mat)
    return (transform_matrix[:3, :3] @ points.T + transform_matrix[:3, [3]]).T


def save_pcd(points, save_pcd_file):
    """ASCII PCD writer"""
    point_num = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 1\n"
        "TYPE F F F U\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {point_num}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_num}\n"
        "DATA ascii\n"
    )
    arr = np.empty_like(points)
    arr[:, :3] = points[:, :3]
    arr[:, 3] = points[:, 3].astype(np.int32)
    np.savetxt(save_pcd_file, arr, fmt="%.6f %.6f %.6f %d", header=header, comments='')


def parse_ext(poses):
    lidar_t = np.array([poses['position']['x'], poses['position']['y'], poses['position']['z']]).reshape(3, 1)
    quat = [poses['heading']['x'], poses['heading']['y'], poses['heading']['z'], poses['heading']['w']]
    lidar_r = R.from_quat(quat).as_matrix()
    lidar_ext = np.hstack((lidar_r, lidar_t))
    lidar_ext = np.vstack((lidar_ext, [0, 0, 0, 1]))
    return lidar_ext


def ensure_dir(input_dir):
    if not exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
    return input_dir


# ✅ robust lidar loader: always return numpy
def get_lidar_frame(seq_name, pandaset_dir, seq, i):
    """Return lidar frame i as numpy array (handles both devkit and manual .pkl)."""
    if len(seq.lidar.data) > i:
        return np.asarray(seq.lidar[i])
    # fallback: manual .pkl load
    lidar_file = os.path.join(pandaset_dir, seq_name, "lidar", f"{i:02d}.pkl")
    with open(lidar_file, "rb") as f:
        frame = pickle.load(f)
    return np.asarray(frame)   # ✅ ensure numpy array


def world_system_precessing(seq, dst_dir, seq_name, pandaset_dir):
    """World coordinate system conversion"""
    dir_map = {
        "front_left_camera": 'camera_image_0', "front_camera": 'camera_image_1',
        "front_right_camera": 'camera_image_2', "right_camera": 'camera_image_3',
        "back_camera": 'camera_image_4', "left_camera": 'camera_image_5'
    }
    dirs = ['lidar_point_cloud_0', 'camera_config'] + list(dir_map.values())
    for _dir in dirs:
        ensure_dir(join(dst_dir, _dir))

    name_num = 0
    n_frames = len(seq.lidar.poses)
    print(f"{seq_name}: {n_frames} lidar poses available")
    for i in tqdm(range(n_frames), desc=f"{seq_name}"):
        name_num += 1
        points = get_lidar_frame(seq_name, pandaset_dir, seq, i)
        pcd_file = join(dst_dir, 'lidar_point_cloud_0', f"{name_num:0>2}.pcd")
        save_pcd(points, pcd_file)

        cam_config = []
        for k, v in dir_map.items():
            if i < len(seq.camera[k].data):
                img = seq.camera[k].data[i]
                img.save(join(dst_dir, v, f"{name_num:0>2}.jpg"))

                cam_pose = seq.camera[k].poses[i]
                cam_ext = parse_ext(cam_pose)
                cam_intrinsics = seq.camera[k].intrinsics
                cam_in = {"fx": cam_intrinsics.fx, "fy": cam_intrinsics.fy,
                          "cx": cam_intrinsics.cx, "cy": cam_intrinsics.cy}
                cfg_data = {"camera_internal": cam_in,
                            "camera_external": inv(cam_ext).flatten().tolist()}
                cam_config.append(cfg_data)

        cfg_file = join(dst_dir, 'camera_config', f"{name_num:0>2}.json")
        with open(cfg_file, 'w', encoding='utf-8') as f:
            json.dump(cam_config, f)

        del points, cam_config
        if i % 5 == 0:
            gc.collect()


def car_system_precessing(seq, dst_dir, seq_name, pandaset_dir):
    """Car (ego) coordinate system conversion"""
    dir_map = {
        "front_left_camera": 'camera_image_0', "front_camera": 'camera_image_1',
        "front_right_camera": 'camera_image_2', "right_camera": 'camera_image_3',
        "back_camera": 'camera_image_4', "left_camera": 'camera_image_5'
    }
    dirs = ['lidar_point_cloud_0', 'camera_config'] + list(dir_map.values())
    for _dir in dirs:
        ensure_dir(join(dst_dir, _dir))

    name_num = 0
    n_frames = len(seq.lidar.poses)
    print(f"{seq_name}: {n_frames} lidar poses available")
    for i in tqdm(range(n_frames), desc=f"{seq_name}"):
        name_num += 1
        points = get_lidar_frame(seq_name, pandaset_dir, seq, i)
        intensity = points[:, 3]
        ego = lidar_points_to_ego(points[:, :3], seq.lidar.poses[i])
        points = np.hstack((ego, intensity.reshape(-1, 1)))
        pcd_file = join(dst_dir, 'lidar_point_cloud_0', f"{name_num:0>2}.pcd")
        save_pcd(points, pcd_file)

        lidar_pose = seq.lidar.poses[i]
        lidar_ext = parse_ext(lidar_pose)
        cam_config = []
        for k, v in dir_map.items():
            if i < len(seq.camera[k].data):
                img = seq.camera[k].data[i]
                img.save(join(dst_dir, v, f"{name_num:0>2}.jpg"))

                cam_pose = seq.camera[k].poses[i]
                cam_ext = parse_ext(cam_pose)
                config_ext = inv(lidar_ext) @ cam_ext
                cam_intrinsics = seq.camera[k].intrinsics
                cam_in = {"fx": cam_intrinsics.fx, "fy": cam_intrinsics.fy,
                          "cx": cam_intrinsics.cx, "cy": cam_intrinsics.cy}
                cfg_data = {"camera_internal": cam_in,
                            "camera_external": inv(config_ext).flatten().tolist()}
                cam_config.append(cfg_data)

        cfg_file = join(dst_dir, 'camera_config', f"{name_num:0>2}.json")
        with open(cfg_file, 'w', encoding='utf-8') as f:
            json.dump(cam_config, f)

        del points, cam_config, lidar_pose, lidar_ext, intensity, ego
        if i % 5 == 0:
            gc.collect()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pandaset_dir', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--is_car_system', default='false', type=str, choices=['true', 'false'],
                        help='Whether to establish a point cloud coordinate system with the acquisition vehicle as the origin')
    args = parser.parse_args()

    pandaset_dir = args.pandaset_dir
    save_dir = args.save_dir
    is_car_system = args.is_car_system
    dataset = DataSet(pandaset_dir)

    selected_sequences = ["016", "017"]

    for seq_name in selected_sequences:
        print(f"Processing sequence {seq_name}")
        dst_dir = join(save_dir, seq_name)
        seq = dataset[seq_name]

        seq.load()
        try:
            seq.lidar.load()
        except Exception:
            print("seq.lidar.load() not implemented, using manual .pkl loader")

        print("Lidar poses:", len(seq.lidar.poses))
        lidar_dir = os.path.join(pandaset_dir, seq_name, "lidar")
        print("Lidar frames available (pkl count):", len([f for f in os.listdir(lidar_dir) if f.endswith('.pkl')]))

        if is_car_system == 'true':
            car_system_precessing(seq, dst_dir, seq_name, pandaset_dir)
        else:
            world_system_precessing(seq, dst_dir, seq_name, pandaset_dir)

        try:
            seq.unload()
        except Exception:
            pass
        gc.collect()
