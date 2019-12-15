#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : show_bboxes.py
#   Author      : YunYang1994
#   Created date: 2019-05-29 01:18:24
#   Description :
#
#================================================================

import cv2
import numpy as np
from PIL import Image, ImageDraw
import colorsys


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def main(label_txt, class_file):
    classes = read_class_names(class_file)
    num_imgs = len(open(label_txt).readlines())
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    fontScale = 0.5
    for img_ind in range(num_imgs):
        image_info = open(label_txt).readlines()[img_ind].split()
        image_path = image_info[0]
        image = cv2.imread(image_path)
        image_h, image_w = image.shape[0], image.shape[1]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        for bbox in image_info[1:]:
            bbox = bbox.split(",")
            class_ind = int(float(bbox[8]))
            coor = np.array([float(c) for c in bbox[:8]], dtype=np.int32)
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2, c3, c4 = (coor[0], coor[1]), (coor[2], coor[3]), (coor[4], coor[5]), (coor[6], coor[7])
            pts = np.array(coor.reshape(4,2))
            cv2.polylines(image, [pts], True, bbox_color)
            # cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
            target_size = (coor[2]-coor[0])*(coor[3]-coor[1])
            print(target_size)
            if target_size < 150 and classes[class_ind]!='person':
                # print(image_path.split('/')[-1])
                print('small target:', target_size, classes[class_ind])
            bbox_mess = '%s' % (classes[class_ind])
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled
            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        image = Image.fromarray(np.uint8(image))
        print(image_path.split('/')[-1])
        image.show()
        pass


if __name__ == '__main__':
    # label_txt = "/home/tamar/DBs/Reccelite/CroppedDB/croppedImgs_1_2_3_5_Th06_reg_rare.txt"
    label_txt = '/home/tamar/DBs/Reccelite/All_data/dataTxt_4points_Tagging1.txt'
    ## file to print target size and target type of input DB (label_txt)
    # f_data_out_path = '/home/tamar/DBs/Reccelite/CroppedDB/individualCropped/5_cropped.txt'

    class_file = '/home/tamar/DBs/Reccelite/All_data/class_names.txt'
    main(label_txt, class_file)