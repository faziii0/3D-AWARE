import pickle
import numpy as np
from mayavi import mlab

# --- File paths ---
lidar_file = "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/001/lidar/00.pkl"
cuboid_file = "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/001/annotations/cuboids/00.pkl"

# --- Load LiDAR ---
with open(lidar_file, 'rb') as f:
    lidar_df = pickle.load(f)   # Pandas DataFrame

x = lidar_df["x"].values
y = lidar_df["y"].values
z = lidar_df["z"].values
i = lidar_df["i"].values  # intensity

# --- Load Cuboids ---
with open(cuboid_file, 'rb') as f:
    cuboids = pickle.load(f)   # Pandas DataFrame

# --- Define colors for labels ---
label_colors = {
    "Car": (1, 0, 0),           # red
    "Truck": (0, 0, 1),         # blue
    "Bus": (0, 1, 0),           # green
    "Pedestrian": (1, 1, 0),    # yellow
    "Bicycle": (1, 0, 1),       # magenta
    "Motorcycle": (0, 1, 1),    # cyan
}

# --- Function to draw cuboid ---
def draw_cuboid(row):
    cx, cy, cz = row["position.x"], row["position.y"], row["position.z"]
    l, w, h = row["dimensions.x"], row["dimensions.y"], row["dimensions.z"]
    yaw = row["yaw"]

    # Choose color by label, default gray
    color = label_colors.get(row["label"], (0.5, 0.5, 0.5))

    # 8 corners of the cuboid
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

    # Rotation (yaw around z-axis)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0, 0, 1]])
    rotated = (R @ corners.T).T
    translated = rotated + np.array([cx, cy, cz])

    # Edges between corners
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    # Draw edges
    for e in edges:
        p1, p2 = translated[e[0]], translated[e[1]]
        mlab.plot3d([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color, tube_radius=None)

# --- Plot LiDAR points ---
mlab.figure(bgcolor=(1,1,1), size=(1000,800))
mlab.points3d(x, y, z, i,
              mode="point",
              colormap="spectral",
              scale_factor=1)

# --- Draw all cuboids ---
for _, row in cuboids.iterrows():
    draw_cuboid(row)

# --- Show interactive window ---
mlab.show()

