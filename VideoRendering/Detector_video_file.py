import cv2, time, os, tensorflow as tf
import numpy as np
import time
import imageio


from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(120)

class Detector:
    def __init__(self):
        pass


    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)

        if(cap.isOpened() == False):
            print("Error opening Video File....")
            return

        (success, image) = cap.read()
        height, width, _ = image.shape

        # Define the output file path for the video
        output_video_path = self.modelName + ".mp4"

        out = imageio.get_writer(output_video_path, fps=30)

        while success:
            bboxImage, car_count, truck_count, pedestrian_count = self.createBoundingBox(image, threshold)
            bboxImage = self.draw_analytics_box(bboxImage, car_count, truck_count, pedestrian_count)

            # Write the annotated image frame to the output video file
            out.append_data(cv2.cvtColor(bboxImage, cv2.COLOR_BGR2RGB))

            (success, image) = cap.read()

            cv2.imshow("Result", bboxImage)

            # Check if user has pressed 'q' to manually stop the process
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


        out.close()
        cap.release()
        cv2.destroyAllWindows()



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


    def createBoundingBox(self, image, threshold = 0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, 
        iou_threshold=threshold, score_threshold=threshold)

        print(bboxIdx)

        car_count = 0
        truck_count = 0
        pedestrian_count = 0

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                # get the class index
                classIndex = classIndexes[i]

                #assign the class label to classLabelText
                classLabelText = self.classesList[classIndex].upper()

                # Count cars and trucks and pedestrians 
                if classLabelText == 'CAR':
                    car_count += 1
                elif classLabelText == 'TRUCK':
                    truck_count += 1
                elif classLabelText == 'PERSON': 
                    pedestrian_count += 1


        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]


                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax) 

                # Get the text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size, _ = cv2.getTextSize(displayText, font, font_scale, font_thickness)

                # Draw a semi-transparent rectangle around the text
                overlay = image.copy()
                rect_x, rect_y, rect_w, rect_h = xmin, ymin - text_size[1] - 15, text_size[0] + 10, text_size[1] + 5
                cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), classColor, -1)
                # alpha is the transparency
                alpha = 0.3 
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                # Draw the bounding box and text
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin + 5, ymin - 10), font, font_scale, (255, 255, 255), font_thickness)

                ###############################
                #These 4 lines add the bold around the edges of the bounding boxes
                lineWidth = min(int((xmax - xmin)*0.2), int((ymax - ymin) * 0.2)) 

                cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=3)
                cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=3)
                cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=3)
                cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=3)

                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=3)
                cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=3)
                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=3)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=3)


        return image, car_count, truck_count, pedestrian_count

    
    def draw_analytics_box(self, image, car_count, truck_count, pedestrian_count):
        imH, imW, imC = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        padding = 10

        # Calculate traffic level
        traffic_level = 'Low'
        traffic_color = (0, 255, 0)  # Green
        if car_count + truck_count > 10:
            traffic_level = 'Moderate'
            traffic_color = (0, 255, 255)  # Yellow
        if car_count + truck_count > 20:
            traffic_level = 'High'
            traffic_color = (0, 0, 255)  # Red

        # Calculate pedestrian density
        pedestrian_density = 'Low'
        pedestrian_color = (0, 255, 0)  # Green
        if pedestrian_count > 10:
            pedestrian_density = 'Moderate'
            pedestrian_color = (0, 255, 255)  # Yellow
        if pedestrian_count > 20:
            pedestrian_density = 'High'
            pedestrian_color = (0, 0, 255)  # Red

        # Define analytics texts
        analytics_texts = [
            f'Cars: {car_count}',
            f'Trucks: {truck_count}',
            f'Pedestrian Count: {pedestrian_count}',
            f'Traffic Level: {traffic_level}',
            f'Pedestrian Density: {pedestrian_density}',
        ]

        # Calculate the size of the box
        text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness) for text in analytics_texts]
        box_width = max(text_size[0][0] for text_size in text_sizes) + 2 * padding
        box_height = sum(text_size[0][1] for text_size in text_sizes) + (len(analytics_texts) + 2) * padding

        # Draw the semi-transparent box
        box_x, box_y = imW - box_width, 0
        overlay = image.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)

        #transperancy
        alpha = 0.8 
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw the texts
        y = box_y + padding
        for i, text in enumerate(analytics_texts):
            text_width, text_height = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            x = box_x + (box_width - text_width) // 2
            y += text_height + padding
            text_color = (255, 255, 255)
            if i == 3:  # Color only the traffic level text
                text_color = traffic_color
            elif i == 4:  # Color only the pedestrian density text
                text_color = pedestrian_color
            cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness)

        return image

            
    def predictImage(self, imagePath, threshold = 0.5):
        image = cv2.imread(imagePath)

        bboxImage = self.createBoundingBox(image, threshold)

        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
