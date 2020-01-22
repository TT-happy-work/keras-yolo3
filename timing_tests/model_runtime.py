"""
Retrain the YOLO model for your own dataset.
"""

import sys
# import tensorflow.keras as keras
# sys.modules['keras']=keras
# import keras.backend as K
# from keras.layers import Input, Lambda
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
# from kerassurgeon.operations import delete_channels
# from kerassurgeon import Surgeon,identify
# from keract import get_activations
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"]=""
# from random import randint
import time
# from tensorflow.python.client import timeline
import tensorflow as tf
import numpy as np
from pruning import get_prunable_layers

def _main():
    annotation_path = '../model_data/cropped.txt'
    log_dir = 'logs/000/'
    classes_path = '../model_data/recce.names'
    anchors_path = '../model_data/recce_anchors_2.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    prunable_layers = get_prunable_layers()
    path_to_frozen_model = '../model_data/frozen_model.pb'
    input_shape = (640, 800)  # multiple of 32, hw

    data_format = 'channels_last'
    # 'channels_first' == NCHW, 'channels_last' = NHWC

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    batch_size = 1
    val_inputs_generator = data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes)
    val_inputs = []
    for i in range(10):
        val_inputs.append(val_inputs_generator.__next__()[0])



    graph = load_graph(path_to_frozen_model)
    ops = []
    for op in graph.get_operations():
        ops.append(op)

    input = graph.get_tensor_by_name('import/input_1:0')
    outputs = [graph.get_tensor_by_name('import/conv2d_75/BiasAdd:0'),
               graph.get_tensor_by_name('import/conv2d_67/BiasAdd:0'),
               graph.get_tensor_by_name('import/conv2d_59/BiasAdd:0')]


    with tf.Session(graph=graph) as sess:
        for i in range(10):
            sess.run(outputs, feed_dict={input: val_inputs[i][0]})

        runtimes = []
        for i in range(10):
            start_time = time.perf_counter()
            sess.run(outputs, feed_dict={input: val_inputs[0][0]})
            end_time = time.perf_counter()
            runtimes.append(end_time-start_time)

    a = 1
#     # Pruning cycles and extra training after each pruning
#     #     data_format = getattr(layer, 'data_format', 'channels_last')

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph

if __name__ == '__main__':
    _main()
