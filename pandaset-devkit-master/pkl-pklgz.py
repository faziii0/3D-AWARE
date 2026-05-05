import os
import gzip
import shutil

# Paths
#lidar_path = "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/043/lidar"
#output_path = "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/043/lidar_gz"


lidar_path = "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/043/annotations/cuboids"
output_path = "/media/bvb/SSD-1/panda/pandaset-devkit-master/data/pandaset/043/annotations/cuboids_gz"

# Make output folder if not exists
os.makedirs(output_path, exist_ok=True)

# Loop through all .pkl files
for fname in os.listdir(lidar_path):
    if fname.endswith(".pkl"):
        pkl_file = os.path.join(lidar_path, fname)
        gz_file = os.path.join(output_path, fname + ".gz")

        # Compress to .pkl.gz
        with open(pkl_file, "rb") as f_in:
            with gzip.open(gz_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"Converted: {pkl_file} -> {gz_file}")

print("✅ Done! All .pkl files compressed into lidar_gz/ (originals kept).")

