import os
from detector_frames_tfrecord import *

#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

classFile = "coco.names"
threshold = 0.3

# Folder containing video files
folder_path = "/mnt/data/videoStorage"

# Get a list of all files in the folder
all_files = os.listdir(folder_path)

# Filter for video files with the .mp4 extension
video_files = [f for f in all_files if f.endswith(".mp4")]

# Find the index of the specified video
start_video = r"Village of Tilton - Traffic Camera 2023-02-08 13_15 [5_XSYlAfJZM].mp4"
start_index = video_files.index(start_video)

detector = Detector(modelURL, classFile)

# Loop through the video files starting from the specified video
for video_file in video_files[start_index:]:
    video_path = os.path.join(folder_path, video_file)
    detector.predictVideo(video_path, threshold)






#videoPath = r"/mnt/data/videoStorage/Sharx Security Demo Live Cam： rotary traffic circle Derry NH USA 2023-02-04 07_47 [fuuBpBQElv4].mp4"

#conda create --name tf_gpu python=3.9
#conda activate tf_gpu2
# conda install install cudatoolkit=11.2 cudnn=8.1 -c=conda-forge
# conda install tensorflow-gpu=2.6 
# import tensorflow as tf
# tf.test.is_gpu_available()
# pip install opencv-python
# C:\Users\14087\Desktop\AppliedProject_CV\AutoAnnotation\Auto-Annotate>"C:\Users\14087\anaconda3\envs\tf_gpu2\python.exe" c:\Users\14087\Desktop\AppliedProject_CV\AutoAnnotation\Auto-Annotate\run_me.py