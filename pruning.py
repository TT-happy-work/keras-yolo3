
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import tensorflow as tf
import keras
from train import high_apoz, get_sample_apoz, get_weakest_channels_in_layer,\
                  data_generator_wrapper,get_anchors,get_classes
from kerassurgeon.operations import delete_channels
from kerassurgeon import Surgeon,identify
from keras.utils import CustomObjectScope
from collections import OrderedDict
import json
from tensorflow.python.client import timeline
from operator import itemgetter
import sys


def get_prunable_layers():

    # The prunable_layers_id  list should be set according to the model built.
    # The keras model names all the convolutional layers as conv2d_{i}, as for now didn't manage naming the tensors
    # according to the part of the net they are in.

    ######################### with original darknet 53  ######################################
    prunable_layers_id = [1, 3, 6, 8, 11,13,15,17,19,21,23,25, # until first skip
                            28, 30, 32, 34, 36, 38, 40,42,     # until second skip
                            45, 47, 49,51,                     # until the end of darknet
                            53, 54, 55, 56, 57,                # until large objs branch
                            60, 61, 62, 63, 64,65,68,69,70,71,72,73,
                            58,66,74                           # conv before objects branches -
                            # objects branches -    59,67,75
                          ]

    ######################### darknet resnet 5 blocks  ######################################
    # prunable_layers_id = [3,4, 7,8 ,11,12, 15,16, 19,20,            # darknet resnet
    #                         22, 23, 24, 25, 26, 28,29, 30,31,32,    # out of the backbone
    #                         33, 34, 35, 37, 38, 39,40, 41,
    #                         42, 43
    #                       ]

    ######################### short list for testing  ######################################
    prunable_layers_id = [3, 6]



    prunable_conv_layers = ['conv2d_{}'.format(v) for v in prunable_layers_id]


    return prunable_conv_layers


# The pruning main should be called each time we want to perform a pruning cycle, which includes:
# 1. Identifying the weakest filters in each prunable layer
# 2. Perform the pruning itself with keras surgeon
# 3. Train the pruned model for some more epochs

# Identifying the weakest filters is done at the function get_weakest_channels_in_layer.
# Currently they are chosen according to APoZ. To set different criteria change only this function


def _main():


    epochs_trained = int(sys.argv[1])
    pruning_cycle = int(sys.argv[2])
    # print('epochs trained is ' + epochs_trained + ' and pruning cycle are' +pruning_cycle)
    annotation_path = 'model_data/cropped.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/recce.names'
    anchors_path = 'model_data/recce_anchors_2.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    recovery_epochs = 1  # TODO - decide number of epochs after pruning
    data_format = 'channels_last'  # 'channels_first' == NCHW, 'channels_last' = NHWC
    input_shape = (640, 800)  # multiple of 32, hw
    # TODO: receive model_path as argument
    model_path = log_dir + 'trained_model_stage_2.h5'


    # model = keras.models.load_model(log_dir + 'trained_model_stage_2.h5', compile=False)
    model = keras.models.load_model(model_path, compile=False)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred},
                  options=run_options, run_metadata=run_metadata)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    terminate_on_NaN = TerminateOnNaN()

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    validation_data = data_generator_wrapper(lines[num_train:], 1, input_shape, anchors, num_classes)

    prunable_layers = get_prunable_layers()
    print('Start Pruning Cycle {}:'.format(pruning_cycle))

    # This section runs some images from the validation set to get the runtimes timeline profiling
    for i in range(10):
    # TODO:
    #  Couldn't open CUDA library libcupti.so.10.0. LD_LIBRARY_PATH:
    #  soultion was to add /usr/local/cuda/extras/CUPTI/lib64 to LD_LIBRARY_PATH in the pycharm configurations
    #  this export is already in bashrc, not sure why it didn't work
        val_input = validation_data.__next__()[0]
        model.predict(val_input, batch_size=None)

    # timeline object for runtime analysis
    tl = timeline.Timeline(step_stats=run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
    f.close()
    # timeline json
    with open('timeline.json') as json_file:
        data = json.load(json_file)

    # the conv_events and conv_dur
    gpu_proccess_id = 5
    # total_dur = 0
    # conv_events = []
    # conv_dur = 0
    conv_layers = {}

    for event in data['traceEvents']:
        if event['pid'] == gpu_proccess_id and 'dur' in event:
                # total_dur += event['dur']
                #if 'name' in event['args'] and 'Conv2D' in event['args']['name']:
                if 'name' in event['args'] and 'conv2d' in event['args']['name']:
                    conv_layer = event['args']['name']
                    # leave next line in case we manage to add the tensor names prefixes in keras
                    # for prefix in prefixes:
                    #     conv_layer = conv_layer.replace(prefix, '')

                    conv_layer, _ = conv_layer.split('/', 1)
                    if conv_layer not in conv_layers:
                        conv_layers[conv_layer] = event['dur']
                    else:
                        conv_layers[conv_layer] += event['dur']
                    # conv_events.append(event)
                    # conv_dur += event['dur']
    sorted_conv_layers = OrderedDict(sorted(conv_layers.items(), key=itemgetter(1), reverse=True))
    # sorted_conv_layers is a dictionary with runtime according to layer.
    # Using this information to set parameters for how much do we want to prune is currently not implemented.

    surgeon = Surgeon(model, copy=True)
    for layer_name in prunable_layers:
        layer = model.get_layer(name=layer_name)
        channels = get_weakest_channels_in_layer(model=model, layer_name=layer_name,
                                                 val_inputs_generator=validation_data, val_set_len=num_val)
        # returns a channels list (if empty don't delete any channel in layer)
        if len(channels) != 0:
            surgeon.add_job('delete_channels', layer, channels=channels)
    # the actual pruning operation happens here:
    model = surgeon.operate()

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change

    batch_size = 2  # note that more GPU memory is required after unfreezing the body
    print('Train for {} more epochs after pruning'.format(recovery_epochs))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=epochs_trained+recovery_epochs,
        initial_epoch=epochs_trained,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])  # original
        # callbacks=[logging, checkpoint, early_stopping, terminate_on_NaN])
        # callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_NaN])
    model.save_weights(log_dir + 'trained_weights_pruning_cycle_{}.h5'.format(pruning_cycle))
    model.save(log_dir + 'trained_model_pruning_cycle_{}.h5'.format(pruning_cycle))
    epochs_trained = epochs_trained + recovery_epochs

    # model.save_weights(log_dir + 'trained_weights_final.h5')
    # model.save(log_dir + 'trained_model_final.h5')

    # EOD - this part performs a pruning cycle of loaded model.
    # now the flow suppose to be running the train script, when reaching the pruning cycles call this script
    # in a while loop and kill it each time to avoid memory issues


if __name__ == '__main__':
    _main()
