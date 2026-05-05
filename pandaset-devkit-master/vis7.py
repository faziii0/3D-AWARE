import pickle
import numpy as np
import pandas as pd
from mayavi import mlab

import sys, numpy
sys.modules["numpy._core"] = numpy.core


# --- File paths (use filtered versions) ---
base_path = "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/001"
lidar_file = f"{base_path}/lidar_fov_good/00.pkl"
cuboid_file = f"{base_path}/annotations/cuboids_fov_good/00.pkl"

# --- Load filtered LiDAR (safe format) ---
with open(lidar_file, "rb") as f:
    records = pickle.load(f)              # list of dicts
lidar_df = pd.DataFrame.from_records(records)

x = lidar_df["x"].values
y = lidar_df["y"].values
z = lidar_df["z"].values
i = lidar_df["i"].values

# --- Load filtered cuboids (safe format) ---
with open(cuboid_file, "rb") as f:
    records = pickle.load(f)              # list of dicts
cuboids = pd.DataFrame.from_records(records)

print("Classes in this frame:", cuboids["new_label"].value_counts())

# --- Colors for simplified labels ---
label_colors = {
    "Vehicle": (1, 0, 0),      # red
    "Pedestrian": (0, 1, 0),   # green
    "Cyclist": (0, 0, 1),      # blue
}

# --- Function to draw cuboid + label ---
def draw_cuboid(row):
    cx, cy, cz = row["position.x"], row["position.y"], row["position.z"]
    l, w, h = row["dimensions.x"], row["dimensions.y"], row["dimensions.z"]
    yaw = row["yaw"]

    color = label_colors.get(row["new_label"], (0.5, 0.5, 0.5))

    # 8 corners
    corners = np.array([
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2]
    ])

    # Rotation (yaw around Z)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0, 0, 1]])
    rotated = (R @ corners.T).T
    translated = rotated + np.array([cx, cy, cz])

    # Edges
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    for e in edges:
        p1, p2 = translated[e[0]], translated[e[1]]
        mlab.plot3d([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color, tube_radius=None)

    # Add text label above cuboid
    mlab.text3d(cx, cy, cz + h/2 + 0.5,
                row["new_label"],
                scale=0.5, color=color)

# --- Plot LiDAR points ---
mlab.figure(bgcolor=(1,1,1), size=(1000,800))
mlab.points3d(x, y, z, i,
              mode="point",
              colormap="spectral",
              scale_factor=1)

# --- Draw cuboids ---
for _, row in cuboids.iterrows():
    draw_cuboid(row)

mlab.title("KITTI-like LiDAR + Simplified Labels", size=0.5)
mlab.view(azimuth=180, elevation=70, distance=100)
mlab.show()

