import tensorflow as tf

feature_description = {
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/confidence': tf.io.VarLenFeature(tf.float32),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    xmin_list = parsed_example['image/object/bbox/xmin'].values.numpy().tolist()
    ymin_list = parsed_example['image/object/bbox/ymin'].values.numpy().tolist()
    xmax_list = parsed_example['image/object/bbox/xmax'].values.numpy().tolist()
    ymax_list = parsed_example['image/object/bbox/ymax'].values.numpy().tolist()

    labels = parsed_example['image/object/class/text'].values.numpy().tolist()
    labels = [label.decode('utf-8') for label in labels]
    confidences = parsed_example['image/object/class/confidence'].values.numpy().tolist()

    image_height = parsed_example['image/height'].numpy()
    image_width = parsed_example['image/width'].numpy()

    print(f"Image dimensions: {image_width}x{image_height}")

    for i, label in enumerate(labels):
        print(f"Bounding box {i + 1}: {label}, confidence: {confidences[i]}")
        print(f"  xmin: {xmin_list[i]}, ymin: {ymin_list[i]}, xmax: {xmax_list[i]}, ymax: {ymax_list[i]}")

    return parsed_example

tfrecord_path = r"/mnt/data/tfrecords/LIVE Port Miami Webcam with VHF Marine Radio Feed from PTZtv 2023-03-30 21_15 [DxZziUUr6CY]/LIVE Port Miami Webcam with VHF Marine Radio Feed from PTZtv 2023-03-30 21_15 [DxZziUUr6CY]_frame_0020.tfrecord"
dataset = tf.data.TFRecordDataset(tfrecord_path)

for raw_example in dataset:
    parsed_example = parse_example(raw_example)
    break
