import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import numpy as np
import colorsys
import random
import cv2


# os.environ["CUDA_VISIBLE_DEVICES"]=""


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except IOError:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def draw_bbox(image, bboxes, classes):
    """
    [[[original function from Yun-Yuan: bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.]]]
    bboxes: [cls_id, probability, x_min, y_min, x_max, y_max] format coordinates.]
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    # random.seed(None)

    # print('len(bboxes) = ', len(bboxes))
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[2:], dtype=np.int32)
        font_scale = 0.5
        score = bbox[1]
        class_ind = int(bbox[0])
        # print('class_ind = ', class_ind)
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        bbox_mess = '%s: %.2f' % (classes[class_ind], score)
        t_size = cv2.getTextSize(bbox_mess, 0, font_scale, thickness=bbox_thick // 2)[0]
        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

        cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def detect_img_to_file(yolo, annotation_file, pred_output_path, write_image_path=None):
    res = []
    with open(annotation_file) as f:
        annotation_lines = f.readlines()
        # shuffle dataset

    for line in annotation_lines:
        line = line.strip('\n')
        image_name = os.path.expanduser(line.split(' ')[0])
        image = Image.open(image_name)
        res = yolo.detect_image_to_file(image).tolist()  # res is all the bb-results(+image path) of a single image
        bboxes_pr = []
        np_res = []
        for bbox in res:  # l is a single bbox in the image
            l_str = []
            np_l = np.array(bbox)
            np_res.append(np_l)
            for i in range(bbox.__len__()):
                l_str.append(str(bbox[i]))
            l_str[0] = yolo.class_names[int(float(l_str[0]))]
            bboxes_pr.append(bbox[2:])
        # c = [[yolo.class_names[int(l[-1])], l[-2], l[:-2]] for l in np_res]
        # write predictions to files (file per img)
        os.makedirs(pred_output_path, exist_ok=True)
        predict_result_path = os.path.join(pred_output_path, image_name.split('/')[-1].split('.')[0] + '.txt')
        with open(predict_result_path, 'w') as f:
            for bbox in np_res:
                coor = np.array(bbox[2:], dtype=np.int32)
                score = bbox[1]
                class_ind = int(bbox[0])
                class_name = yolo.class_names[int(class_ind)]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())

        # with open(os.path.join(pred_output_path, image_name.split('/')[-1].split('.')[0] +'.txt'), 'w') as f:
        #     [f.write(" ".join(x) + "\n") for x in res]
        # save img with bboxes drawn
        if write_image_path:
            os.makedirs(write_image_path, exist_ok=True)
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = draw_bbox(image, np_res, yolo.class_names)
            im_path = os.path.join(write_image_path, image_name.split('/')[-1])
            cv2.imwrite(im_path, image)

    yolo.close_session()
    return res


def parse_commandline():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    yolo_def = argparse.Namespace(**YOLO.get_defaults())

    '''
    Command line options
    '''
    parser.add_argument('--model', type=str, help='path to model weight file, default %s' % yolo_def.model_path)

    parser.add_argument('--anchors', type=str, help='path to anchor definitions, default %s' % yolo_def.anchors_path)

    parser.add_argument('--classes', type=str, help='path to class definitions, default %s' % yolo_def.classes_path)

    parser.add_argument('--gpu_num', type=int, help='Number of GPU to use, default %s' % yolo_def.gpu_num)

    parser.add_argument('--image', default=False, action="store_true",
                        help='Image detection mode, will ignore all positional arguments')

    parser.add_argument('--val_annotation_file', type=str, help='file of validation-set files')

    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument("--input", nargs='?', type=str, required=False,
                        help="Video input path")

    parser.add_argument("--output", nargs='?', type=str, default="",
                        help="[Optional] Video output path")

    parser.add_argument("--write_image_path", type=str, required=False,
                        help="path to save image with predicted bboxes on it")

    opts = parser.parse_args()
    return opts, parser


if __name__ == '__main__':
    options, parser = parse_commandline()
    if options.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in options:
            print(" Ignoring remaining command line arguments: " + options.input + "," + options.output)
        detect_img(YOLO(**vars(options)))
    elif "input" in options:
        detect_video(YOLO(**vars(options)), options.input, options.output)
    elif "val_annotation_file" in options:
        detect_img_to_file(YOLO(**vars(options)), options.val_annotation_file, options.output, options.write_image_path)
    else:
        print("Must specify at least video_input_path. See usage with --help.")
