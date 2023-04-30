import cv2, time, os, tensorflow as tf
import numpy as np
import time
import keyboard
import xml.etree.cElementTree as ET
from collections import defaultdict
import shutil
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.train import Example, Features, Feature, BytesList, FloatList, Int64List
from tensorflow.io import serialize_tensor, TFRecordWriter

np.random.seed(123)


class Detector:

    def __init__(self, modelURL, classFile):
        self.downloadModel(modelURL)
        self.loadModel()
        self.readClasses(classFile)
        self.bboxes = []


    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)

        if(cap.isOpened() == False):
            print("Error opening Video File....")
            return

        (success, image) = cap.read()
        height, width, _ = image.shape

        # Set the output directory for TFRecords
        output_annotations_dir = "/mnt/data/tfrecords"
        os.makedirs(output_annotations_dir, exist_ok=True)

        frame_count = 0

        while success:
            print(f"Processing frame {frame_count}")
            original_frame = image.copy()  # Save a copy of the original frame
            bboxImage = self.createBoundingBox(image, threshold)

            # Convert the annotations to tf.train.Example format
            # Save the frame as a temporary file
            
            temp_frame_file = "temp_frame_{}.jpg".format(frame_count)
            cv2.imwrite(temp_frame_file, original_frame)

            # Pass the temporary file path to the create_tf_example function
            annotated_frame = self.create_tf_example(temp_frame_file, self.bboxes, height, width)

            # Remove the temporary file
            os.remove(temp_frame_file)

            # Write the example to a separate TFRecord file
            video_name = os.path.splitext(os.path.basename(videoPath))[0]
            annotation_file = os.path.join(output_annotations_dir, f"{video_name}_frame_{frame_count:04d}.tfrecord")
            with tf.io.TFRecordWriter(annotation_file) as tfrecord_writer:
                tfrecord_writer.write(annotated_frame.SerializeToString())
            print(f"Finished writing TFRecord for frame {frame_count}")

            print(f"Number of detections: {len(self.bboxes)}")

            self.bboxes = []
            (success, image) = cap.read()

            cv2.imshow("Result", bboxImage)  # Display the frame with bounding boxes
            key = cv2.waitKey(1) & 0xFF  # Wait for a key press and mask the result

            if key == ord('q'):  # Check if the pressed key is 'q'
                print('Stopping video processing manually...')
                break

            frame_count += 1

        # Organize the TFRecords after they are created
        organize_tfrecords(output_annotations_dir)

        # Move the processed video to another folder
        processed_video_dir = "/mnt/data/videosParsed"
        os.makedirs(processed_video_dir, exist_ok=True)
        video_name = os.path.basename(videoPath)
        shutil.move(videoPath, os.path.join(processed_video_dir, video_name))

        cap.release()
        cv2.destroyAllWindows()


    def create_tf_example(self, image_path, bboxes, image_height, image_width):
        """
        Convert the bounding boxes to tf.train.Example format
        """
        encoded_bboxes = []
        encoded_labels = []
        encoded_scores = []

        for box in bboxes:
            encoded_labels.append(box[0])
            encoded_scores.append(box[1])
            encoded_bboxes.extend([box[2], box[3], box[4], box[5]])

        # Read the image file
        with open(image_path, "rb") as image_file:
            encoded_image_data = image_file.read()

        # Create the Example record and include the encoded image data
        feature = Features(feature={
            'image/encoded': Feature(bytes_list=BytesList(value=[encoded_image_data])),
            'image/height': Feature(int64_list=Int64List(value=[image_height])),
            'image/width': Feature(int64_list=Int64List(value=[image_width])),
            'image/object/bbox/xmin': Feature(float_list=FloatList(value=encoded_bboxes[0::4])),
            'image/object/bbox/ymin': Feature(float_list=FloatList(value=encoded_bboxes[1::4])),
            'image/object/bbox/xmax': Feature(float_list=FloatList(value=encoded_bboxes[2::4])),
            'image/object/bbox/ymax': Feature(float_list=FloatList(value=encoded_bboxes[3::4])),
            'image/object/class/text': Feature(bytes_list=BytesList(value=[label.encode('utf-8') for label in encoded_labels])),
            'image/object/class/confidence': Feature(float_list=FloatList(value=encoded_scores)),
        })

        return Example(features=feature)
    
    def createBoundingBox(self, image, threshold=0.5):
        self.bboxes = []  # Clear the list before adding new bounding boxes

        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=70,
                                            iou_threshold=threshold, score_threshold=threshold)

        print(bboxIdx)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, color=classColor, thickness=1)


                self.bboxes.append([classLabelText, classConfidence, xmin, ymin, xmax, ymax])

        return image

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

            #colors list
            self.colorList =np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

            print(len(self.classesList), len(self.colorList))

    def downloadModel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

        print(fileName)
        print(self.modelName)

        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName,
        origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))

        print("Model " + self.modelName + " loaded successfully....")


    def predictImage(self, imagePath, threshold = 0.5):
        image = cv2.imread(imagePath)

        bboxImage = self.createBoundingBox(image, threshold)

        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def organize_tfrecords(output_annotations_dir):
    def get_group_key(file):
        # Extract the title before the second "_" delimiter
        return "_".join(file.split("_")[:2])

    file_groups = defaultdict(list)

    for file in os.listdir(output_annotations_dir):
        if file.endswith('.tfrecord'):
            group_key = get_group_key(file)
            file_groups[group_key].append(file)

    for group_key, files in file_groups.items():
        group_folder = os.path.join(output_annotations_dir, group_key)

        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        for file in files:
            src_path = os.path.join(output_annotations_dir, file)
            dst_path = os.path.join(group_folder, file)
            shutil.move(src_path, dst_path)

