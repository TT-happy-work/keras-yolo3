"""
Retrain the YOLO model for your own dataset.
"""

import sys
# import tensorflow.keras as keras
# sys.modules['keras']=keras
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from kerassurgeon.operations import delete_channels
from kerassurgeon import Surgeon,identify
from keract import get_activations
# from pruning import get_prunable_layers
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
from random import randint
import time
from tensorflow.python.client import timeline
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import gc

def _main():
    annotation_path = '../model_data/cropped.txt'
    log_dir = 'logs/000/'
    classes_path = '../model_data/recce.names'
    anchors_path = '../model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    # prunable_layers = get_prunable_layers()
    path_to_frozen_model = '../model_data/frozen_model.pb'
    data_format = 'channels_first'
    # 'channels_first' == NCHW, 'channels_last' = NHWC

    input_shape = (640, 800)  # multiple of 32, hw

    model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, load_pretrained=False, data_format=data_format) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)

    # model.save(filepath=path_to_keras_model)

    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "../model_data/", "frozen_model.pb", as_text=False)


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


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5', data_format='channels_last'):
    '''create the training model'''
    K.clear_session() # get a new session
    #nadav_wp
    # image_input = Input(shape=(None, None, 3))
    # h, w = input_shape
    h, w = input_shape
    if data_format == 'channels_last':
        image_input = Input(shape=(h, w, 3))
    elif data_format == 'channels_first':
        image_input = Input(shape=(3, h, w))
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes, data_format=data_format)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    # nadav_wp_runtime-
    # original
    # model = Model([model_body.input, *y_true], model_loss)
    # changed
    model = Model(model_body.input, model_body.output)
    # -nadav_wp_runtime

    return model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


if __name__ == '__main__':
    _main()
