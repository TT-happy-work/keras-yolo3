"""
Retrain the YOLO model for your own dataset.
"""

import sys
# import tensorflow.keras as keras
# sys.modules['keras']=keras
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from kerassurgeon.operations import delete_channels
from kerassurgeon import Surgeon, identify
from keract import get_activations
# from pruning import get_prunable_layers
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
from random import randint
import time
from tensorflow.python.client import timeline
import tensorflow as tf
import json
from collections import OrderedDict
from operator import itemgetter
from keras.utils import CustomObjectScope


def _main():
    annotation_path = 'model_data/cropped.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/recce.names'
    anchors_path = 'model_data/recce_anchors_2.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    weights_path_tiny = 'model_data/tiny_yolo_weights.h5'
    weights_path_nominal = 'model_data/yolo_weights_pony.h5'
    data_format = 'channels_last'  # 'channels_first' == NCHW, 'channels_last' = NHWC
    pruning_cycle = 0
    epochs_trained = 0
    first_stage_epochs = 1  # 50
    second_stage_epochs = 1  # 200
    # after_pruning_epochs = 1  # TODO - decide number of epochs after pruning

    pre_pruning_training = False  # perform training before pruning. set to false if already loads train model for pruning
    perform_pruning_cycle = True

    input_shape = (640, 800)  # multiple of 32, hw

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    terminate_on_NaN = TerminateOnNaN()

    val_split = 0.1  # the part of the dataset that goes to the validation set
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if pre_pruning_training:    # the usual training without pruning, includes the model creation and loading weights
        is_tiny_version = len(anchors) == 6  # default setting
        if is_tiny_version:
            model = create_tiny_model(input_shape, anchors, num_classes,
                                      freeze_body=2, weights_path=weights_path_tiny)
        else:
            model = create_model(input_shape, anchors, num_classes,
                                 freeze_body=2, weights_path=weights_path_nominal, load_pretrained=True,
                                 data_format=data_format)  # make sure you know what you freeze

        # this options are relevant only for timeline profiling, we moved the timing tests elsewhere
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
        if True:
            model.compile(optimizer=Adam(lr=1e-3), loss={
                # use custom yolo_loss Lambda layer.
                'yolo_loss': lambda y_true, y_pred: y_pred})

            batch_size = 8
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(
                data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                       num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=first_stage_epochs,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
            model.save_weights(log_dir + 'trained_weights_stage_1.h5')
            model.save(log_dir + 'trained_model_stage_1.h5')

        epochs_trained = first_stage_epochs

        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if True:
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=1e-4),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = 2  # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(
                data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                       num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=epochs_trained + second_stage_epochs,
                initial_epoch=epochs_trained,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])  # original
            # callbacks=[logging, checkpoint, early_stopping, terminate_on_NaN])
            # callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_NaN])
            model.save_weights(log_dir + 'trained_weights_stage_2.h5')
            model.save(log_dir + 'trained_model_stage_2.h5')

        epochs_trained = epochs_trained + second_stage_epochs
    # Further training if needed.

    # Pruning cycles and extra training after each pruning:

    while(perform_pruning_cycle):
        pruning_cycle = pruning_cycle + 1
        os.system('python pruning.py ' + str(epochs_trained) + ' ' + str(pruning_cycle))


        # TODO: decide when switching perform_pruning_cycle off



    #     model.save_weights(log_dir + 'trained_weights_final.h5')
    #     model.save(log_dir + 'trained_model_final.h5')


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
    K.clear_session()  # get a new session
    h, w = input_shape
    if data_format == 'channels_last':
        image_input = Input(shape=(h, w, 3))
    elif data_format == 'channels_first':
        image_input = Input(shape=(3, h, w))
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    # original -
    model = Model([model_body.input, *y_true], model_loss)
    # changed - uncomment next line instead of the previous to set the model output to be only the convolutions
    # model = Model(model_body.input, model_body.output)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    h, w = input_shape
    image_input = Input(shape=(h, w, 3))
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def high_apoz(apoz, method="std", cutoff_std=1, cutoff_absolute=0.99):
    """
    Args:
        apoz: List of the APoZ values for each channel in the layer.
        method: Cutoff method for high APoZ. "std", "absolute" or "both".
        cutoff_std: Channels with a higher APoZ than the layer mean plus
            `cutoff_std` standard deviations will be identified for pruning.
        cutoff_absolute: Channels with a higher APoZ than `cutoff_absolute`
            will be identified for pruning.
    Returns:
        high_apoz_channels: List of indices of channels with high APoZ.
    """
    if method not in {'std', 'absolute', 'both'}:
        raise ValueError('Invalid `mode` argument. '
                         'Expected one of {"std", "absolute", "both"} '
                         'but got', method)
    if method == "std":
        cutoff = apoz.mean() + apoz.std() * cutoff_std
    elif method == 'absolute':
        cutoff = cutoff_absolute
    else:
        cutoff = min([cutoff_absolute, apoz.mean() + apoz.std() * cutoff_std])

    return np.where(apoz >= cutoff)[0]


def get_sample_apoz(activations):
    """ get APoZ for filters in layer according to given
    The APoZ a.k.a. (A)verage (P)ercentage (o)f activations equal to (Z)ero,
        activations: flattened layer activation map with shape (filter output dimension, number of filters)
    Returns:
        ndarray of APoZ values for each channel in the layer.
    """

    # Flatten all except channels axis
    activations = np.reshape(activations, [-1, activations.shape[-1]])
    zeros_loc = (activations == 0).astype(int)
    curr_apoz = np.sum(zeros_loc, axis=0) / activations.shape[0]

    return curr_apoz


def get_weakest_channels_in_layer(model, layer_name, val_inputs_generator, val_set_len):
    """ get the channels that will be pruned, the ones with lowest contribution in the layer.
    we rank the contribution of a channel based on a criteria (APoZ in our case) calculated on the output feature map
    of the layer and normalized over the validation set.

    Returns:
        list of channels with lowest contribution in the layer.
    """
    layer = model.get_layer(name=layer_name)
    data_format = getattr(layer, 'data_format', 'channels_last')
    num_channels = layer.filters
    apoz = np.zeros(num_channels)

    # apoz parameters
    cutoff_method = 'absolute'  # possible values are'std', 'absolute', 'both'
    cutoff_std = 1
    cutoff_absolute = 0.4

    for i in range(val_set_len):
        val_input = val_inputs_generator.__next__()[0]
        # returns an activations dictionary (key based on tensor name)
        activations = get_activations(model=model, x=val_input, layer_name=layer_name)
        # activations = activations[layer_name+'/convolution:0']
        # this lines are a fix to find tensor names, which change during pruning cycles
        activations = [val for key, val in activations.items() if layer_name in key]
        activations = activations[0]

        if data_format == 'channels_first':  # Ensure that the channels axis is last
            activations = np.swapaxes(activations, 1, -1)

        apoz = apoz + get_sample_apoz(activations)
    apoz = apoz / val_set_len  # normalize according to given validation set size

    # at this point we have the apoz for each channel averaged over the validation dataset
    # high apoz function from keras surgeon decides which channels are considered to have high apoz, they will be cut
    channels = high_apoz(apoz, cutoff_method, cutoff_std, cutoff_absolute)

    return channels


if __name__ == '__main__':
    _main()















# worked with loop of pruning, caused OOM issues


# """
# Retrain the YOLO model for your own dataset.
# """
#
# import sys
# # import tensorflow.keras as keras
# # sys.modules['keras']=keras
# import numpy as np
# import keras.backend as K
# from keras.layers import Input, Lambda
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
#
# from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
# from yolo3.utils import get_random_data
#
# from kerassurgeon.operations import delete_channels
# from kerassurgeon import Surgeon,identify
# from keract import get_activations
# from pruning import get_prunable_layers
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"]=""
# from random import randint
# import time
# from tensorflow.python.client import timeline
# import tensorflow as tf
# import json
# from collections import OrderedDict
# from operator import itemgetter
#
#
# def _main():
#     annotation_path = 'model_data/cropped.txt'
#     log_dir = 'logs/000/'
#     classes_path = 'model_data/recce.names'
#     anchors_path = 'model_data/recce_anchors_2.txt'
#     class_names = get_classes(classes_path)
#     num_classes = len(class_names)
#     anchors = get_anchors(anchors_path)
#     weights_path_tiny = 'model_data/tiny_yolo_weights.h5'
#     weights_path_nominal = 'model_data/yolo_weights_pony.h5'
#     first_stage_epochs = 1   # 50
#     second_stage_epochs = 1  # 200
#     after_pruning_epochs = 1   # TODO - decide number of epochs after pruning
#
#     data_format = 'channels_last'  # 'channels_first' == NCHW, 'channels_last' = NHWC
#
#     input_shape = (640, 800) # multiple of 32, hw
#
#     is_tiny_version = len(anchors)==6 # default setting
#     if is_tiny_version:
#         model = create_tiny_model(input_shape, anchors, num_classes,
#             freeze_body=2, weights_path=weights_path_tiny)
#     else:
#         model = create_model(input_shape, anchors, num_classes,
#             freeze_body=2, weights_path=weights_path_nominal, load_pretrained=True, data_format=data_format) # make sure you know what you freeze
#                                                             # nadav_wp_darknet33- original load_pretrained = true
#
#     logging = TensorBoard(log_dir=log_dir)
#     checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
#         monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
#     terminate_on_NaN = TerminateOnNaN()
#
#     val_split = 0.1
#     with open(annotation_path) as f:
#         lines = f.readlines()
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     num_val = int(len(lines)*val_split)
#     num_train = len(lines) - num_val
#
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#
#     model.compile(options=run_options, run_metadata=run_metadata, optimizer=Adam(lr=1e-3), loss={
#             # use custom yolo_loss Lambda layer.
#             'yolo_loss': lambda y_true, y_pred: y_pred})
#
#     # times_before_pruning_after_compile = []
#     # for i in range(10):
#     #     model.predict(val_inputs[i][0])
#     # for i in range(10):
#     #     start = time.perf_counter()
#     #     model.predict(val_inputs[0][0])
#     #     end = time.perf_counter()
#     #     times_before_pruning_after_compile.append(end-start)
#     #
#     # model.predict(val_inputs[0][0])
#     # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
#     # with open('timeline.ctf.json', 'w') as f:
#     #     f.write(trace.generate_chrome_trace_format())
#
#
#     # surgeon = Surgeon(model, copy=False)
#     # for layer_name in prunable_layers:
#     #     layer = model.get_layer(name=layer_name)
#     #     num_channels = layer.filters
#     #     # channels = [i+1 for i in range(randint(0,3))]
#     #     channels = [i + 1 for i in range(int(num_channels * 0.4))]
#     #     while len(channels) == 0:
#     #         channels = [i + 1 for i in range(randint(0, 4))]
#     #     surgeon.add_job('delete_channels', layer, channels=channels)
#     #
#     # model_pruned = surgeon.operate()
#     # model_pruned.compile()
#
#
#     # layer = model.get_layer(name='conv2d_1')
#     # val_inputs_generator = data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes)
#     # val_inputs = []
#     # for i in range(num_val):
#     #     val_inputs.append(val_inputs_generator.__next__()[0])
#     # # not working - because of branching of the outputs the model don't reach the end
#     # apoz = identify.get_apoz(model, layer, val_inputs[0][0])
#     # high_apoz_channels = identify.high_apoz(apoz)
#     # channels = [2, 3]
#     # model_pruned = delete_channels(model, layer, channels)
#     # -nadav_wp_pruning
#
#
#     # Train with frozen layers first, to get a stable loss.
#     # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
#     if True:
#         model.compile(optimizer=Adam(lr=1e-3), loss={
#             # use custom yolo_loss Lambda layer.
#             'yolo_loss': lambda y_true, y_pred: y_pred})
#
#         batch_size = 8
#         print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
#         model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
#                 steps_per_epoch=max(1, num_train//batch_size),
#                 validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
#                 validation_steps=max(1, num_val//batch_size),
#                 epochs=first_stage_epochs,
#                 initial_epoch=0,
#                 callbacks=[logging, checkpoint])
#         model.save_weights(log_dir + 'trained_weights_stage_1.h5')
#         model.save(log_dir + 'trained_model_stage_1.h5')
#
#     epochs_trained = first_stage_epochs
#     # Unfreeze and continue training, to fine-tune.
#     # Train longer if the result is not good.
#     if True:
#         for i in range(len(model.layers)):
#             model.layers[i].trainable = True
#         model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
#         print('Unfreeze all of the layers.')
#
#         batch_size = 2 # note that more GPU memory is required after unfreezing the body
#         print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
#         model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
#             steps_per_epoch=max(1, num_train//batch_size),
#             validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
#             validation_steps=max(1, num_val//batch_size),
#             epochs=epochs_trained + second_stage_epochs,
#             initial_epoch=epochs_trained,
#             callbacks=[logging, checkpoint, reduce_lr, early_stopping])  # original
#             # callbacks=[logging, checkpoint, early_stopping, terminate_on_NaN])
#             # callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_NaN])
#         model.save_weights(log_dir + 'trained_weights_stage_2.h5')
#         model.save(log_dir + 'trained_model_stage_2.h5')
#
#     epochs_trained = epochs_trained + second_stage_epochs
#     # Further training if needed.
#     # Pruning cycles and extra training after each pruning
#
#     finished_pruning = False
#     pruning_cycle = 0
#     validation_data = data_generator_wrapper(lines[num_train:], 1, input_shape, anchors, num_classes)
#     # TODO - set a goal image runtime or detection results low boundary to check at the end of an iteration to see if we finished pruning
#
#     while not finished_pruning:
#         # EOD - OOM issues with pruning.
#         # this loop should be in a different script with the format of
#         # 1. load weights
#         # 2.while loop of looping/training
#         # 3.saving the model
#
#         prunable_layers = get_prunable_layers()
#         #####
#         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#         run_metadata = tf.RunMetadata()
#         model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred},
#                       options=run_options, run_metadata=run_metadata)
#         # Run model in your usual way
#         for i in range(10):
#             # TODO: -
#             #  Couldn't open CUDA library libcupti.so.10.0. LD_LIBRARY_PATH:
#             #  soultion was to add /usr/local/cuda/extras/CUPTI/lib64 to LD_LIBRARY_PATH in the pycharm configurations
#             #  this export is already in bashrc, not sure why it didn't work
#             val_input = validation_data.__next__()[0]
#             model.predict(val_input, batch_size=None)
#
#         ######## timeline object for runtime analysis #######
#         tl = timeline.Timeline(step_stats=run_metadata.step_stats)
#         ctf = tl.generate_chrome_trace_format()
#         with open('timeline.json', 'w') as f:
#             f.write(ctf)
#         f.close()
#         ####### timeline json #######
#         with open('timeline.json') as json_file:
#             data = json.load(json_file)
#
#         # TODO - training with online runtime analysis
#         gpu_proccess_id = 5
#         # total_dur = 0
#         # conv_events = []
#         # conv_dur = 0
#         conv_layers = {}
#
#         for event in data['traceEvents']:
#             if event['pid'] == gpu_proccess_id and 'dur' in event:
#                     # total_dur += event['dur']
#                     #if 'name' in event['args'] and 'Conv2D' in event['args']['name']:
#                     if 'name' in event['args'] and 'conv2d' in event['args']['name']:
#                         conv_layer = event['args']['name']
#                         # for prefix in prefixes:
#                         #     conv_layer = conv_layer.replace(prefix, '')
#
#                         # conv_layer, _ = conv_layer.split('Conv2D', 1)
#                         # conv_layer, _ = conv_layer.split('conv2d', 1)
#                         conv_layer, _ = conv_layer.split('/', 1)
#                         if conv_layer not in conv_layers:
#                             conv_layers[conv_layer] = event['dur']
#                         else:
#                             conv_layers[conv_layer] += event['dur']
#                         # conv_events.append(event)
#                         # conv_dur += event['dur']
#         sorted_conv_layers = OrderedDict(sorted(conv_layers.items(), key=itemgetter(1), reverse=True))
#
#         pruning_cycle = pruning_cycle + 1
#         print('Start Pruning Cycle {}:'.format(pruning_cycle))
#         surgeon = Surgeon(model, copy=False)
#         # EOD - need to see if setting it to false solve OOM problems. The problem with this was that
#         # keract get_activations after pruning didn't work. need to see why
#         # surgeon = Surgeon(model, copy=True)
#         for layer_name in prunable_layers:
#             layer = model.get_layer(name=layer_name)
#             channels = get_weakest_channels_in_layer(model=model, layer_name=layer_name,
#                                                      val_inputs_generator=validation_data, val_set_len=num_val)
#             # returns a channels list (if empty don't delete any channel in layer)
#             if len(channels) != 0:
#                 surgeon.add_job('delete_channels', layer, channels=channels)
#         model = surgeon.operate()
#
#         for i in range(len(model.layers)):
#             model.layers[i].trainable = True
#         model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
#
#         batch_size = 2  # note that more GPU memory is required after unfreezing the body
#         print('Train for {} more epochs after pruning'.format(after_pruning_epochs))
#         model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
#             steps_per_epoch=max(1, num_train//batch_size),
#             validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
#             validation_steps=max(1, num_val//batch_size),
#             epochs=epochs_trained+after_pruning_epochs,
#             initial_epoch=epochs_trained,
#             callbacks=[logging, checkpoint, reduce_lr, early_stopping])  # original
#             # callbacks=[logging, checkpoint, early_stopping, terminate_on_NaN])
#             # callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_NaN])
#         model.save_weights(log_dir + 'trained_weights_pruning_cycle_{}.h5'.format(pruning_cycle))
#         model.save(log_dir + 'trained_model_pruning_cycle_{}.h5'.format(pruning_cycle))
#         epochs_trained = epochs_trained + after_pruning_epochs
#
#         # TODO:
#         # create new runtimes json
#         # update finished_pruning flag according to runtimes or detection results
#
#     model.save_weights(log_dir + 'trained_weights_final.h5')
#     model.save(log_dir + 'trained_model_final.h5')
#
#
# def get_classes(classes_path):
#     '''loads the classes'''
#     with open(classes_path) as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#     return class_names
#
# def get_anchors(anchors_path):
#     '''loads the anchors from a file'''
#     with open(anchors_path) as f:
#         anchors = f.readline()
#     anchors = [float(x) for x in anchors.split(',')]
#     return np.array(anchors).reshape(-1, 2)
#
#
# def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
#             weights_path='model_data/yolo_weights.h5', data_format='channels_last'):
#     '''create the training model'''
#     K.clear_session() # get a new session
#     #nadav_wp
#     # image_input = Input(shape=(None, None, 3))
#     # h, w = input_shape
#     h, w = input_shape
#     if data_format == 'channels_last':
#         image_input = Input(shape=(h, w, 3))
#     elif data_format == 'channels_first':
#         image_input = Input(shape=(3, h, w))
#     num_anchors = len(anchors)
#
#     y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
#         num_anchors//3, num_classes+5)) for l in range(3)]
#
#     model_body = yolo_body(image_input, num_anchors//3, num_classes)
#     print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
#
#     if load_pretrained:
#         model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
#         print('Load weights {}.'.format(weights_path))
#         if freeze_body in [1, 2]:
#             # Freeze darknet53 body or freeze all but 3 output layers.
#             num = (185, len(model_body.layers)-3)[freeze_body-1]
#             for i in range(num): model_body.layers[i].trainable = False
#             print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
#
#     model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
#         arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
#         [*model_body.output, *y_true])
#     # nadav_wp_runtime-
#     # original
#     model = Model([model_body.input, *y_true], model_loss)
#     # changed - model output is convolution output
#     # model = Model(model_body.input, model_body.output)
#     # -nadav_wp_runtime
#
#     return model
#
#
# def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
#             weights_path='model_data/tiny_yolo_weights.h5'):
#     '''create the training model, for Tiny YOLOv3'''
#     K.clear_session() # get a new session
#     #nadav_wp
#     # image_input = Input(shape=(None, None, 3))
#     # h, w = input_shape
#     h, w = input_shape
#     image_input = Input(shape=(h, w, 3))
#     num_anchors = len(anchors)
#
#     y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
#         num_anchors//2, num_classes+5)) for l in range(2)]
#
#     model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
#     print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
#
#     if load_pretrained:
#         model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
#         print('Load weights {}.'.format(weights_path))
#         if freeze_body in [1, 2]:
#             # Freeze the darknet body or freeze all but 2 output layers.
#             num = (20, len(model_body.layers)-2)[freeze_body-1]
#             for i in range(num): model_body.layers[i].trainable = False
#             print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
#
#     model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
#         arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
#         [*model_body.output, *y_true])
#     model = Model([model_body.input, *y_true], model_loss)
#
#     return model
#
#
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
#
#
# def high_apoz(apoz, method="std", cutoff_std=1, cutoff_absolute=0.99):
#     """
#     Args:
#         apoz: List of the APoZ values for each channel in the layer.
#         method: Cutoff method for high APoZ. "std", "absolute" or "both".
#         cutoff_std: Channels with a higher APoZ than the layer mean plus
#             `cutoff_std` standard deviations will be identified for pruning.
#         cutoff_absolute: Channels with a higher APoZ than `cutoff_absolute`
#             will be identified for pruning.
#     Returns:
#         high_apoz_channels: List of indices of channels with high APoZ.
#     """
#     if method not in {'std', 'absolute', 'both'}:
#         raise ValueError('Invalid `mode` argument. '
#                          'Expected one of {"std", "absolute", "both"} '
#                          'but got', method)
#     if method == "std":
#         cutoff = apoz.mean() + apoz.std()*cutoff_std
#     elif method == 'absolute':
#         cutoff = cutoff_absolute
#     else:
#         cutoff = min([cutoff_absolute, apoz.mean() + apoz.std()*cutoff_std])
#
#     return np.where(apoz >= cutoff)[0]
#
#
# def get_sample_apoz(activations):
#     """ get APoZ for filters in layer according to given
#     The APoZ a.k.a. (A)verage (P)ercentage (o)f activations equal to (Z)ero,
#         activations: flattened layer activation map with shape (filter output dimension, number of filters)
#     Returns:
#         ndarray of APoZ values for each channel in the layer.
#     """
#
#     # Flatten all except channels axis
#     activations = np.reshape(activations, [-1, activations.shape[-1]])
#     zeros_loc = (activations == 0).astype(int)
#     curr_apoz = np.sum(zeros_loc, axis=0) / activations.shape[0]
#
#     return curr_apoz
#
#
# def get_weakest_channels_in_layer(model, layer_name, val_inputs_generator,val_set_len):
#     channels =[]
#     layer = model.get_layer(name=layer_name)
#     data_format = getattr(layer, 'data_format', 'channels_last')
#     num_channels = layer.filters
#     apoz = np.zeros(num_channels)
#
#     # apoz parameters
#     cutoff_method = 'absolute'  # possible values are'std', 'absolute', 'both'
#     cutoff_std = 1
#     cutoff_absolute = 0.4
#
#     for i in range(val_set_len):
#         val_input = val_inputs_generator.__next__()[0]
#         # returns an activations dictionary (key based on tensor name)
#         activations = get_activations(model=model, x=val_input, layer_name=layer_name)
#         # activations = activations[layer_name+'/convolution:0']
#         # this lines are a fix to find tensor names, which change during pruning cycles
#         activations = [val for key, val in activations.items() if layer_name in key]
#         activations = activations[0]
#
#
#         if data_format == 'channels_first':  # Ensure that the channels axis is last
#             activations = np.swapaxes(activations, 1, -1)
#
#         apoz = apoz + get_sample_apoz(activations)
#     apoz = apoz/val_set_len  # normalize according to given validation set size
#
#     # at this point we have the apoz for each channel averaged over the validation dataset
#     # high apoz function from keras surgeon decides which channels are considered to have high apoz, they will be cut
#     channels = high_apoz(apoz, cutoff_method, cutoff_std, cutoff_absolute)
#     # TODO: remove next line
#     channels = [0,1]
#
#     return channels
#
#
# if __name__ == '__main__':
#     _main()
