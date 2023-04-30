import cv2, time, os, tensorflow as tf
import numpy as np
import time
import keyboard
import xml.etree.cElementTree as ET
from PIL import Image
from tensorflow.python.keras.utils.data_utils import get_file

#this takes the image and creates annotations


np.random.seed(123)

class Detector:
    def __init__(self):
        pass
        

    def writeAnnotations(self, annotation_file, imagePath, bboxes, imageWidth, imageHeight):
        print("writeAnnotations")
        #imageWidth = 1440
        #imageHeight = 1080
        print(imagePath)
        print(annotation_file)

        with open(f"{imagePath}_annotation_file.xml", 'w') as f:
            f.write('<annotation>\n')
            f.write('\t<folder>Auto-Annotate</folder>\n')
            f.write('\t<filename>{}</filename>\n'.format(os.path.basename(imagePath)))
            f.write('\t<path>{}</path>\n'.format(imagePath))
            f.write('\t<source>\n')
            f.write('\t\t<database>Unknown</database>\n')
            f.write('\t</source>\n')
            f.write('\t<size>\n')
            f.write('\t\t<width>{}</width>\n'.format(imageWidth))
            f.write('\t\t<height>{}</height>\n'.format(imageHeight))
            f.write('\t\t<depth>3</depth>\n')
            f.write('\t</size>\n')
            f.write('\t<segmented>0</segmented>\n')
            for bbox in bboxes:
                f.write('\t<object>\n')
                f.write('\t\t<name>{}</name>\n'.format(bbox[0]))
                f.write('\t\t<pose>Unspecified</pose>\n')
                f.write('\t\t<truncated>0</truncated>\n')
                f.write('\t\t<difficult>0</difficult>\n')
                f.write('\t\t<bndbox>\n')
                f.write('\t\t\t<xmin>{}</xmin>\n'.format(bbox[1]))
                f.write('\t\t\t<ymin>{}</ymin>\n'.format(bbox[2]))
                f.write('\t\t\t<xmax>{}</xmax>\n'.format(bbox[3]))
                f.write('\t\t\t<ymax>{}</ymax>\n'.format(bbox[4]))
                f.write('\t\t</bndbox>\n')
                f.write('\t</object>\n')
            f.write('</annotation>')




    def createBoundingBox(self, imagePath, image, threshold=0.5):
        print("createBoundingBox")
        print(image)
        self.bboxes = []  # Clear the list before adding new bounding boxes

        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                            iou_threshold=threshold, score_threshold=threshold)


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

        # Write the annotations to the file
        annotation_path = self.modelName + "_annotations.xml"
        self.writeAnnotations(annotation_path, imagePath, self.bboxes, imW, imH)

        return image, classLabelText, classConfidence, xmin, ymin, xmax, ymax

    def predictImage(self, imagePath, threshold = 0.5):
        print("predictImage")
        print(imagePath)

        image = cv2.imread(imagePath)


        bboxImage = self.createBoundingBox(imagePath, image, threshold)
        bboxImage = cv2.cvtColor(bboxImage[0].astype('uint8'), cv2.COLOR_RGB2BGR)


        #cv2.imwrite(self.modelName + ".jpg", bboxImage)

        #cv2.imshow("Result", bboxImage)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Convert bboxImage to numpy array, flatten it, and reshape it
        bboxImage = np.array(bboxImage).ravel()
        bboxImage = bboxImage.reshape((bboxImage.shape[0] // 3, 3))
        return bboxImage


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


    def writeAnnotations2(self, annotation_file, bboxes):
        # Create the root element
        annotation = ET.Element("annotation")
        
        for box in bboxes:
            # Convert the bounding box coordinates to integers
            box = [x for x in box]
            classLabelText = box[0]
            classConfidence = box[1]
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            
            object_elem = ET.SubElement(annotation, 'object')  # create object element
            name_elem = ET.SubElement(object_elem, 'name')  # create name element
            name_elem.text = classLabelText.upper()  # set text of name element
            confidence_elem = ET.SubElement(object_elem, 'confidence')  # create confidence element
            confidence_elem.text = str(classConfidence)  # set text of confidence element
            bndbox_elem = ET.SubElement(object_elem, 'bndbox')  # create bndbox element
            xmin_elem = ET.SubElement(bndbox_elem, 'xmin')  # create xmin element
            xmin_elem.text = str(xmin)  # set text of xmin element
            ymin_elem = ET.SubElement(bndbox_elem, 'ymin')  # create ymin element
            ymin_elem.text = str(ymin)  # set text of ymin element
            xmax_elem = ET.SubElement(bndbox_elem, 'xmax')  # create xmax element
            xmax_elem.text = str(xmax)  # set text of xmax element
            ymax_elem = ET.SubElement(bndbox_elem, 'ymax')  # create ymax element
            ymax_elem.text = str(ymax)  # set text of ymax element
            
        # create ElementTree object and write to file
        tree = ET.ElementTree(annotation)
        with open(annotation_file.name, 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)


    def writeAnnotations3(self,annotation_file, imagePath, bboxes):
        print("writeAnnotations")
        #annotation_file = os.path.abspath(annotation_file)
        print(annotation_file)

        root = ET.Element("annotation")
        folder = ET.SubElement(root, "folder")
        folder.text = "Auto-Annotate"

        filename = ET.SubElement(root, "filename")
        filename.text = imagePath

        path = ET.SubElement(root, "path")
        path.text = imagePath

        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        width=800
        height=1333

        print(width, height)
        
        size = ET.SubElement(root, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(height)
        depth_elem = ET.SubElement(size, "depth")
        depth_elem.text = "3"  # Assume RGB image

        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"

        for bbox in bboxes:
            obj = ET.SubElement(root, "object")
            name = ET.SubElement(obj, "name")
            name.text = bbox[0]
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            ymin = ET.SubElement(bndbox, "ymin")
            xmax = ET.SubElement(bndbox, "xmax")
            ymax = ET.SubElement(bndbox, "ymax")
            xmin.text = (bbox[1])
            ymin.text = (bbox[2])
            xmax.text = (bbox[3])
            ymax.text = (bbox[4])

        tree = ET.ElementTree(root)
        with open(annotation_file, 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)