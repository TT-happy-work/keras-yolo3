import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from yolo3.model import yolo_body, yolo_loss
from kerassurgeon import Surgeon
from pruning import get_prunable_layers
import tensorflow as tf
import time
# from tensorflow.python.client import timeline
from PIL import Image, ImageOps
from math import ceil
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.compiler.tensorrt import trt_convert

def _main():
    annotation_path = './model_data/miniDB_full_scale.txt'
    log_dir = 'logs/000/'
    classes_path = './model_data/recce.names'
    anchors_path = './model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    path_to_frozen_model = './model_data/frozen_model.pb'
    data_format = 'channels_first'  # 'channels_first' == NCHW, 'channels_last' = NHWC
    full_scale_img_shape = (2482, 3304)
    patch_shape = (20*32, 25*32) #(1280, 1664) # multiple of 32, hw
    # patch_shape = (30*32, 30*32) #(2496, 3328) # multiple of 32, hw
    # patch_shape = tuple(dim//32*32 for dim in full_scale_img_shape) #(2464, 3296) # multiple of 32, hw
    # tensor names without pruning:
    input_tensor_name = 'input_1:0'
    output_tensors_names = ['conv2d_75/BiasAdd:0',
                            'conv2d_67/BiasAdd:0',
                            'conv2d_59/BiasAdd:0']
    # tensor names with pruning with surgeon (with arg copy == false):
    # input_tensor_name = 'input_1:0'
    # output_tensors_names = ['conv2d_75_1/BiasAdd:0',
    #                         'conv2d_67_1/BiasAdd:0',
    #                         'conv2d_59_1/BiasAdd:0']
    # tensor names with pruning with surgeon (with arg copy == true):
    # input_tensor_name = 'input_1_2:0'
    # output_tensors_names = ['conv2d_75_3/BiasAdd:0',
    #                         'conv2d_67_3/BiasAdd:0',
    #                         'conv2d_59_3/BiasAdd:0']

    create_new_model = True
    perform_rt = False  # need to know the output tensors names if True
    prune_model = False
    pruning_percent = 0.0
    pruning_copy_model = False

    # validate args:
    if create_new_model and perform_rt:
        print("Warning: Can't perform Tensor-RT optimization along with create_new_model - perform_rt will be set to False.")
        perform_rt = False

    # create new model pb file
    if create_new_model:
        print("\n".join(["*******************************",
                         "* Creating a new pb model ... *",
                         "*******************************"]))
        create_model_pb(input_shape=patch_shape, anchors=anchors, num_classes=num_classes,
                        data_format=data_format, log_dir=log_dir, prune_model=prune_model,
                        pruning_percent=pruning_percent, pruning_copy_model=pruning_copy_model)

    #TODO: tensor rt on pb file

    # step 1 - load graph from pb file, get input and output tensors
    # graph = load_graph(path_to_frozen_model)
    print("\n".join(["*************************************************",
                     "* Loading existing / new pb model from file ... *",
                     "*************************************************"]))
    with tf.gfile.GFile(path_to_frozen_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    if perform_rt:
        print("\n".join(["**************************************************",
                         "* Optimizing the loaded graph with Tensor-RT ... *",
                         "**************************************************"]))
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.67)
        graph_def = trt.create_inference_graph(
            input_graph_def=graph_def,
            outputs=output_tensors_names,
            max_batch_size=1,
            precision_mode=trt_convert.TrtPrecisionMode.FP32,
            #minimum_segment_size=50,
            max_workspace_size_bytes=1<<30,
            # is_dynamic_op=True
            # maximum_cached_engines = maximum_cached_engines,
            # cached_engine_batches = cached_engine_batches,
            # input_saved_model_dir = input_saved_model_dir,
            # input_saved_model_tags = input_saved_model_tags,
            # output_saved_model_dir = output_saved_model_dir,
            # session_config = session_config
            )
        print("number of TRT ops: %s" % len([1 for n in graph_def.node if str(n.op)=='TRTEngineOp']))


    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    #

    # used this to find the output tensors name.
    # ops = []
    # for op in graph.get_operations():
    #     ops.append(op)

    # TODO - look for a way to find tensor names automatically
    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    outputs_tensors = [graph.get_tensor_by_name(name) for name in output_tensors_names]

    # with tf.gfile.GFile(path_to_frozen_model, "rb") as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())

    # step 2 - create images list with full scale (might be padded to get a multiplication of 32)
    # step 3 - create from each full scale image list of cropped images with requested patch size

    ph, pw = patch_shape                # ph - patch height, pw - patch width
    ih, iw = full_scale_img_shape
    num_patch_height = ceil(ih/float(ph))   # number of patches in height
    num_patch_width  = ceil(iw/float(pw))   # number of patches in width
    # avoid overlap patches:
    num_patch_height = 1   # number of patches in height
    num_patch_width  = 1   # number of patches in width
    num_patches      = num_patch_height * num_patch_width
    batch_size = 1 # num_patches
    full_scale_imgs = []
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    for annotation_line in annotation_lines:
        line = annotation_line.split()
        image = Image.open(line[0])
        dw = iw - pw
        dh = ih - ph
        border = (dw//2, dh//2, iw-dw//2-pw, ih-dh//2-ph)  # (left, top, right, bottom)
        image = ImageOps.crop(image, border)
        assert image.size == (pw,ph)
        full_scale_imgs.append(np.array(image)/255.)
    cropped_imgs = []
    cropped_img_shape=(num_patches, ph, pw, 3) if data_format == 'channels_last' else (num_patches, 3, ph, pw)
    for img in full_scale_imgs:
        cropped_img = np.empty(shape=cropped_img_shape, dtype=np.float64)
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
            rand_batch = np.random.ranf(cropped_imgs[i].shape)
            sess.run(outputs_tensors, feed_dict={input_tensor: rand_batch})#[:batch_size]})

        
        for batch_size in [1,2,3]:
            print ('running with batch size: %s ...' % batch_size)
            num_batches = len(cropped_imgs) // batch_size
            batches = np.concatenate(cropped_imgs)
            batch_runtimes = []
            #for cropped_img_batch in cropped_imgs:
            for i in range(num_batches):
                batch = batches[i*batch_size:(i+1)*batch_size]
                batch_start_time = time.perf_counter()
                ret = sess.run(outputs_tensors, feed_dict={input_tensor: batch})
                batch_end_time = time.perf_counter()
                batch_runtimes.append(batch_end_time-batch_start_time)
                ret_sizes = [r.shape for r in ret]
                if i == 0:
                    print("Input tensor shapes: ", batch.shape)
                    print("Returned tensor shapes: %s" % ret_sizes)
        
            batch_runtimes = np.array(batch_runtimes) * 1000  # convert to millis

            # print timing statistics:
            print('num_batches is: %s ' % num_batches)
            print('num_images in a batch (bathch_size): %s' % batch_size)
            print('num processed pixels in a batch (bathch_size * image-size): %s' % (batch_size * batches[0].size / 3))
            n_pxls = batch_size * batches[0].size / 3
            m_pxls = n_pxls / 1024 / 1024
            print('num of input mega pixels in a batch: %.03f' % (m_pxls))
            print('per batch timings (millis):    min=%.03f, max=%.03f, avg=%.03f, std=%.03f.' % 
                (batch_runtimes.min(), batch_runtimes.max(), batch_runtimes.mean(), batch_runtimes.std()))
            batch_runtimes /= 1000 # convert back to seconds
            print('Mega-Pixels per second (net):  best=%.03f, worst=%.03f, avg=%.03f.' % 
                (m_pxls/batch_runtimes.min(), m_pxls/batch_runtimes.max(), m_pxls/batch_runtimes.mean()))
    #     cropped_imgs_runtimes = []
    #     for img in cropped_imgs:
    #         batch_runtimes = []
    #         for i in range(batches_per_full_image):
    #             batch = img[i*batch_size:(i+1)*batch_size]
    #             batch_start_time = time.perf_counter()
    #             sess.run(outputs_tensors, feed_dict={input_tensor: batch})
    #             batch_end_time = time.perf_counter()
    #             batch_runtimes.append(batch_end_time-batch_start_time)
    #         cropped_imgs_runtimes.append(np.sum(batch_runtimes))
    #     # a = 1


# model creation functions

def create_model_pb(input_shape, anchors, num_classes, data_format, log_dir,
                    prune_model, pruning_percent, pruning_copy_model):
    model = create_model(input_shape, anchors, num_classes,
                         freeze_body=2, load_pretrained=False,
                         data_format=data_format)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)

    if prune_model == False:
        frozen_graph = freeze_session(K.get_session(),
                                    output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, "./model_data/", "frozen_model.pb", as_text=False)
        return

    # reach here if the model needs to be pruned
    surgeon = Surgeon(model, copy=pruning_copy_model)
    prunable_layers = get_prunable_layers()
    for layer_name in prunable_layers:
        layer = model.get_layer(name=layer_name)
        num_channels = layer.filters
        channels = [i + 1 for i in range(int(num_channels * pruning_percent))]
        surgeon.add_job('delete_channels', layer, channels=channels)

    ## TODO: 1. check the surgeon with channels_first - done
    ##       2. check name of input tensor after pruning - done
    ##       3. predict on patches one at a time/predict on batch of patches - done
    ##       4. tensor RT on pb file
    ##       5. check new create_pb with pruning on nisko

    model_pruned = surgeon.operate()
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model_pruned.outputs])
    tf.train.write_graph(frozen_graph, "./model_data/", "frozen_model.pb", as_text=False)


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
    K.set_learning_phase(0)
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
        freeze_var_names = list({v.op.name for v in tf.global_variables()}.difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# model runtime functions


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