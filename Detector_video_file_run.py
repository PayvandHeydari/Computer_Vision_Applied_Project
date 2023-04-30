#from DetectorLiveBoxing import *
from Detector_video_file import *

#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

classFile = "coco.names"
#imagePath = "/home/payvand/Desktop/AppliedProject/AppliedProject/AutoAnnotation/Auto-Annotate/images/12283150_12d37e6389_z.jpg"

#imagePath = r"C:\Users\14087\Desktop\AppliedProject_CV\AutoAnnotation\Auto-Annotate\images\Sharx Security Demo Live Cam rotary traffic circle Derry NH USA 2023-02-14 10_40 [fuuBpBQElv4].mp4_25.png"
videoPath = r"/mnt/data/videoStorage/Midway International Airport, Chicago, IL ｜ StreamTime LIVE 2023-03-30 08_28 [S26YOMAD290].mp4"
threshold = 0.3

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
#detector.predictImage(imagePath, threshold)

detector.predictVideo(videoPath, threshold)

#videoPath = r"/mnt/data/videoStorage/LIVE Port Miami Webcam with VHF Marine Radio Feed from PTZtv 2023-04-03 08_18 [DxZziUUr6CY].mp4"


#git remote add origin https://github.com/PayvandHeydari/AppliedProject2.git git branch -M main git push -u origin main
#C:\Users\14087\Desktop\AppliedProject_CV\AutoAnnotation\Auto-Annotate\run_me.py
#conda create --name tf_gpu python=3.9
#conda activate tf_gpu2
# conda install install cudatoolkit=11.2 cudnn=8.1 -c=conda-forge
# conda install tensorflow-gpu=2.6 
# import tensorflow as tf
# tf.test.is_gpu_available()
# pip install opencv-python
# C:\Users\14087\Desktop\AppliedProject_CV\AutoAnnotation\Auto-Annotate>"C:\Users\14087\anaconda3\envs\tf_gpu2\python.exe" c:\Users\14087\Desktop\AppliedProject_CV\AutoAnnotation\Auto-Annotate\run_me.py