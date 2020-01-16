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
from PIL import Image
from math import ceil

def _main():
    annotation_path = '../model_data/miniDB_full_scale.txt'
    # log_dir = 'logs/000/'
    # classes_path = '../model_data/recce.names'
    # anchors_path = '../model_data/yolo_anchors.txt'
    # class_names = get_classes(classes_path)
    # num_classes = len(class_names)
    # anchors = get_anchors(anchors_path)
    # prunable_layers = get_prunable_layers()
    patch_shape = (864, 864)
    full_scale_img_shape = (2482, 3304)
    path_to_frozen_model = '../model_data/frozen_model.pb'
    data_format = 'channels_last'  # 'channels_first' == NCHW, 'channels_last' == NHWC

    # step 1 - load graph from pb file, get input and output tensors
    graph = load_graph(path_to_frozen_model)
    ops = []
    for op in graph.get_operations():
        ops.append(op)
    # when did this after pruning with surgeon (with arg copy == false) tensor names were:
    # input_tensor = graph.get_tensor_by_name('import/input_1:0')
    # outputs = [graph.get_tensor_by_name('import/conv2d_75_1/BiasAdd:0'),
    #            graph.get_tensor_by_name('import/conv2d_67_1/BiasAdd:0'),
    #            graph.get_tensor_by_name('import/conv2d_59_1/BiasAdd:0')]
    # when did this after pruning with surgeon (with arg copy == true) tensor names were:
    # input_tensor = graph.get_tensor_by_name('import/input_1_2:0')
    # outputs = [graph.get_tensor_by_name('import/conv2d_75_3/BiasAdd:0'),
    #            graph.get_tensor_by_name('import/conv2d_67_3/BiasAdd:0'),
    #            graph.get_tensor_by_name('import/conv2d_59_3/BiasAdd:0')]
    # TODO - look for a way to find tensor names automatically
    input_tensor = graph.get_tensor_by_name('import/input_1:0')
    outputs = [graph.get_tensor_by_name('import/conv2d_75/BiasAdd:0'),
               graph.get_tensor_by_name('import/conv2d_67/BiasAdd:0'),
               graph.get_tensor_by_name('import/conv2d_59/BiasAdd:0')]

    # step 2 - create images list with full scale
    with open(annotation_path) as f:
        annotation_lines = f.readlines()

    full_scale_imgs = []
    for annotation_line in annotation_lines:
        line = annotation_line.split()
        image = Image.open(line[0])
        full_scale_imgs.append(np.array(image)/255.)

    # step 3 - create from each full scale image list of cropped images with requested patch size
    ph, pw = patch_shape                # ph - patch height, pw - patch width
    ih, iw = full_scale_img_shape
    num_patch_height = ceil(ih/float(ph))   # number of patches in height
    num_patch_width  = ceil(iw/float(pw))   # number of patches in width
    num_patches      = num_patch_height * num_patch_width
    batch_size = num_patches


    cropped_imgs = []
    for img in full_scale_imgs:
        if data_format == 'channels_last':
            cropped_img = np.empty(shape=(num_patches, ph, pw, 3), dtype=np.float64)
        elif data_format == 'channels_first':
            cropped_img = np.empty(shape=(num_patches, 3, ph, pw), dtype=np.float64)

        cnt=0
        for i in range(num_patch_height):
            for j in range(num_patch_width):
                if ((i+1)*ph < ih) and ((j+1)*pw < iw):
                    patch = img[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
                elif ((i+1)*ph < ih) and ((j+1)*pw > iw):
                    patch = img[i*ph:(i+1)*ph, -pw:, :]
                elif ((i+1)*ph > ih) and ((j+1)*pw < iw):
                    patch = img[-ph:, j*pw:(j+1)*pw, :]
                else:
                    patch = img[-ph:, -pw:, :]

                if data_format == 'channels_last':
                    cropped_img[cnt] = patch
                elif data_format == 'channels_first':
                    cropped_img[cnt] = np.moveaxis(patch, -1, 0)
                cnt = cnt+1
        cropped_imgs.append(cropped_img)


    # step 4 - divide each img to batches and predict on batch
    # TODO: for now all batches have the same size (number of patches is divisible by batch size)
    batches_per_full_image = ceil(num_patches / float(batch_size))

    with tf.Session(graph=graph) as sess:
        # run few images to init the net
        for i in range(len(cropped_imgs)):
            # sess.run(outputs, feed_dict={input_tensor: np.expand_dims(cropped_imgs[i][0], axis=0)})
            sess.run(outputs, feed_dict={input_tensor: cropped_imgs[i][:batch_size]})

        cropped_imgs_runtimes = []
        for img in cropped_imgs:
            batch_runtimes = []
            for i in range(batches_per_full_image):
                batch = img[i*batch_size:(i+1)*batch_size]
                batch_start_time = time.perf_counter()
                sess.run(outputs, feed_dict={input_tensor: batch})
                batch_end_time = time.perf_counter()
                batch_runtimes.append(batch_end_time-batch_start_time)
            cropped_imgs_runtimes.append(np.sum(batch_runtimes))

    a = 1

# def get_classes(classes_path):
#     '''loads the classes'''
#     with open(classes_path) as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#     return class_names
#
#
# def get_anchors(anchors_path):
#     '''loads the anchors from a file'''
#     with open(anchors_path) as f:
#         anchors = f.readline()
#     anchors = [float(x) for x in anchors.split(',')]
#     return np.array(anchors).reshape(-1, 2)


# def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
#     '''data generator for fit_generator'''
#     n = len(annotation_lines)
#     i = 0
#     while True:
#         image_data = []
#         box_data = []
#         for b in range(batch_size):
#             if i==0:
#                 np.random.shuffle(annotation_lines)
#             image, box = get_random_data(annotation_lines[i], input_shape, random=True)
#             image_data.append(image)
#             box_data.append(box)
#             i = (i+1) % n
#         image_data = np.array(image_data)
#         box_data = np.array(box_data)
#         y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
#         yield [image_data, *y_true], np.zeros(batch_size)
#
#
# def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
#     n = len(annotation_lines)
#     if n==0 or batch_size<=0: return None
#     return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

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
