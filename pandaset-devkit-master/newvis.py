import pandaset
import pandas as pd
import numpy as np
from mayavi import mlab

# Load sequence + LiDAR
dataset = pandaset.Dataset("/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset")
seq002 = dataset['002']
seq002.load_lidar()

# Concatenate LiDAR frames (first 5 frames as example)
selected_data = seq002.lidar[:5]
selected_data = [
    xy.assign(f=i)   # add frame index as column f
    for i, xy in enumerate(selected_data)
]
selected_data = pd.concat(selected_data)

# Normalize f column to [0,1] for colors
selected_data['f'] = (selected_data['f'] - selected_data['f'].min()) / (selected_data['f'].max() - selected_data['f'].min())

# Convert to numpy
points = selected_data[['x','y','z']].to_numpy()
colors = selected_data['f'].to_numpy()

# Plot with mayavi
mlab.figure("LiDAR Concatenated", bgcolor=(0,0,0), size=(1000,800))
mlab.points3d(points[:,0], points[:,1], points[:,2],
              colors,
              mode="point", colormap="spectral")  # or "jet", "viridis"

mlab.show()

