import os
import shutil
from collections import defaultdict

# Set your source directory where the TFRecord files are stored
source_dir = "/mnt/data/tfrecords"

# Set your destination directory where the grouped files will be moved
dest_dir = "/mnt/data/tfrecords"

# Create a function to define the grouping criteria
def get_group_key(file):
    # Extract the title before the second "_" delimiter
    return "_".join(file.split("_")[:2])

# Group the files
file_groups = defaultdict(list)

for file in os.listdir(source_dir):
    if file.endswith('.tfrecord'):
        group_key = get_group_key(file)
        file_groups[group_key].append(file)

# Move the files to their respective group folders
for group_key, files in file_groups.items():
    group_folder = os.path.join(dest_dir, group_key)

    if not os.path.exists(group_folder):
        os.makedirs(group_folder)

    for file in files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(group_folder, file)
        shutil.move(src_path, dst_path)
