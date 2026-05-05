import pickle
import numpy as np
from mayavi import mlab

# --- Safe numpy unpickler ---
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        return super().find_class(module, name)

# --- File ---
lidar_file = "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/001/lidar_fov/00.pkl"

with open(lidar_file, "rb") as f:
    points = NumpyUnpickler(f).load()

print("Loaded points:", points.shape)

# --- Ensure correct shape ---
points = np.array(points)
if points.ndim != 2 or points.shape[1] < 3:
    raise ValueError("Unexpected lidar shape:", points.shape)

x, y, z = points[:, 0], points[:, 1], points[:, 2]

# --- Visualization ---
mlab.figure(bgcolor=(0, 0, 0), size=(1000, 800))

# Use spheres instead of "point" for visibility
mlab.points3d(
    x, y, z,
    mode="sphere",
    color=(0, 1, 0),      # green
    scale_factor=0.2      # much larger points
)

# Force camera to center and zoom
mlab.view(azimuth=180, elevation=70, distance=150, focalpoint=[np.mean(x), np.mean(y), np.mean(z)])

mlab.orientation_axes()
mlab.show()

