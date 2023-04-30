import tensorflow as tf
import os
import numpy as np
from PIL import Image
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.core import losses
from object_detection.core import box_list
from object_detection.metrics import coco_evaluation

train_files = r"/home/lab/Desktop/CV/Auto-Annotate/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8/annotations"
val_files = r"/home/lab/Desktop/CV/Auto-Annotate/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8/testing_annotations"
batch_size = 5

pipeline_config_path = '/home/lab/Desktop/CV/Tensorflow/models/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=True)
detection_model = detection_model.as_default()

def _parse_function(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/confidence': tf.io.VarLenFeature(tf.float32),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.image.decode_png(example['image/encoded'], channels=3)

    # Combine the bounding box and class label information
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    class_text = tf.sparse.to_dense(example['image/object/class/text'], default_value='')
    class_confidence = tf.sparse.to_dense(example['image/object/class/confidence'])

    # Process the label data as needed
    label = {'bbox': tf.stack([xmin, ymin, xmax, ymax], axis=-1),
             'class_text': class_text,
             'class_confidence': class_confidence}

    return image, label


def load_dataset(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    return dataset


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.localization_loss = losses.WeightedSmoothL1LocalizationLoss()
        self.classification_loss = losses.WeightedSigmoidClassificationLoss()

    def call(self, y_true, y_pred):
        num_boxes = tf.shape(y_true['bbox'])[1]
        boxlists_true = box_list.BoxList(y_true['bbox'][:, :, :4])
        boxlists_true.add_field('classes', y_true['bbox'][:, :, 4:])
        boxlists_pred = box_list.BoxList(y_pred['bbox'][:, :, :4])
        boxlists_pred.add_field('classes', y_pred['bbox'][:, :, 4:])

        localization_loss = self.localization_loss(boxlists_true, boxlists_pred)
        classification_loss = self.classification_loss(boxlists_true, boxlists_pred)
        normalizer = tf.cast(num_boxes, dtype=tf.float32)

        return (localization_loss + classification_loss) / normalizer

NUM_CLASSES = 10
detection_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=CustomLoss())

epochs = 10
history = detection_model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

eval_tfrecords = "/home/lab/Desktop/CV/Auto-Annotate/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8/annotations"
eval_files = [os.path.join(eval_tfrecords, f) for f in os.listdir(eval_tfrecords) if f.endswith('.tfrecord')]
eval_dataset = load_dataset(eval_files, batch_size)

coco_evaluator = coco_evaluation.CocoDetectionEvaluator(num_groundtruth_classes=NUM_CLASSES)

image_id = 0  # Initialize image_id
for batch_images, batch_labels in eval_dataset:
    batch_predictions = detection_model.predict(batch_images)

    # Convert class labels from text to integer indices
    batch_labels['class_indices'] = tf.map_fn(lambda cls: class_label_to_index[cls.numpy().decode('utf-8')],
                                              batch_labels['class_text'], dtype=tf.int32)

    # Prepare ground truth information
    gt_boxes = np.array(batch_labels['bbox'])
    gt_classes = np.array(batch_labels['class_indices'])
    gt_scores = np.ones_like(gt_classes, dtype=np.float32)

    # Prepare prediction information
    pred_boxes = np.array(batch_predictions['bbox'])
    pred_classes = np.array(batch_predictions['class_indices'])
    pred_scores = np.array(batch_predictions['class_confidence'])

    # Add ground truth and prediction information to the coco_evaluator
    coco_evaluator.add_single_detected_image_info(image_id, {'bbox': np.column_stack((pred_boxes, pred_scores, pred_classes))})
    coco_evaluator.add_single_ground_truth_image_info(image_id, {'bbox': gt_boxes, 'scores': gt_scores, 'classes': gt_classes})

    image_id += 1

coco_evaluator.evaluate()
coco_metrics = coco_evaluator.get_results()
print(coco_metrics)
