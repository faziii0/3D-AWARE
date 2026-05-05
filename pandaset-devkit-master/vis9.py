
import numpy as np

def load_pcd_ascii(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip header lines until "DATA ascii"
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("DATA"):
            start_idx = i + 1
            break

    # Load the point data
    points = np.loadtxt(lines[start_idx:])
    return points   # shape (N, 4) → [x, y, z, intensity]



from mayavi import mlab

# Load point cloud directly
points = load_pcd_ascii(
    "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/save/016/lidar_point_cloud_0/01.pcd"
)

x, y, z, intensity = points[:,0], points[:,1], points[:,2], points[:,3]

# Mayavi 3D scatter
mlab.figure(bgcolor=(1,1,1), size=(800,600))
mlab.points3d(
    x, y, z,
    intensity,                # color by intensity
    mode="point", colormap="spectral", scale_factor=1
)
mlab.show()

