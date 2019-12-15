import os
import cv2
import math
import argparse
import numpy as np
from core import utils


def main(write_image_path, f_data_in_path, f_data_out_path, patch_w, patch_h, cropTH):
    classes = utils.read_class_names("./data/classes/recce.names")
    rare_cats = [list(classes.values()).index('motorcycle'), list(classes.values()).index('truck'),
                 list(classes.values()).index('bus'), list(classes.values()).index('van'),
                 list(classes.values()).index('pickup'), list(classes.values()).index('pickup_noload'),
                 list(classes.values()).index('ambulance'), list(classes.values()).index('suv'),
                 list(classes.values()).index('jeep')]

    if not os.path.exists(write_image_path):
        os.makedirs(write_image_path)
    with open(f_data_in_path, 'r') as f_in:
        txt = f_in.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    new_line_flag = False
    # loop over each img in original list: crop each img and add the cropped_img txt to the new datatxt
    with open(f_data_out_path, 'w') as f_out:
        for img_ind in range(len(annotations)):
            annotation = annotations[img_ind]
            line = annotation.split()
            image_path = line[0]
            if not os.path.exists(image_path):
                raise KeyError("%s does not exist ... " % image_path)
            print(image_path)
            bboxes = np.array([list(map(float, box.split(','))) for box in line[1:]])
            image = np.array(cv2.imread(image_path))
            #print("original image size is:", image.shape)
            image_format = image_path.split('.')[-1]
            image_name = image_path.split('.')[0].split('/')[-1]
            orig_image_h = image.shape[0]
            orig_image_w = image.shape[1]
            patch_counter = 0
            for the_box_ind in range(len(bboxes)):
                if bboxes[the_box_ind][-1] in rare_cats:
                    # print(the_box_ind, bboxes[the_box_ind])
                    new_img_flag = True
                    cropped_bboxes = []
                    cropped_image_name = image_name + '_' + str(patch_counter) + '_' + classes[bboxes[the_box_ind][-1]] + '.' + image_format
                    patch_counter += 1
                    the_box_center = [(bboxes[the_box_ind][0]+bboxes[the_box_ind][2])/2, (bboxes[the_box_ind][1]+bboxes[the_box_ind][3])/2]
                    # box is well within image. then center the patch around target
                    if the_box_center[0]>patch_w/2 and the_box_center[1]>patch_h/2 and the_box_center[0]<orig_image_w-patch_w/2 and the_box_center[1]<orig_image_h-patch_h/2:
                        minW = the_box_center[0]-patch_w/2; minH = the_box_center[1]-patch_h/2
                        maxW = the_box_center[0]+patch_w/2; maxH = the_box_center[1]+patch_h/2
                    # box is on one or two edges of image.
                    else:
                        # take care of width
                        if the_box_center[0] < patch_w/2: # near left edge
                            minW = 0
                            maxW = patch_w
                            if maxW-minW!=800:
                                print('case1, shape=', maxW-minW)
                        elif the_box_center[0] > orig_image_w-patch_w/2: # near right edge
                            maxW = orig_image_w
                            minW = maxW - patch_w
                            if maxW - minW != 800:
                                print('case2, shape=', maxW-minW)
                        else: # horizontally well-centered
                            minW = round(the_box_center[0]) - patch_w/2
                            maxW = round(the_box_center[0]) + patch_w/2
                            if maxW - minW != 800:
                                print('case3, shape=', maxW-minW)
                        # take care of height
                        if the_box_center[1] < patch_h/2: # near top edge
                            minH = 0
                            maxH = patch_h
                            #print('case4, shape=', maxH-minH)
                        elif the_box_center[1] > orig_image_h - patch_h / 2:  # near bottom edge
                            maxH = orig_image_h
                            minH = maxH - patch_h
                            #print('case5, shape=', maxH-minH)
                        else:  # vertically well-centered
                            minH = round(the_box_center[1]) - patch_h/2
                            maxH = round(the_box_center[1]) + patch_h/2
                            #print('case6, shape=', maxH-minH)
                    # congrats. you now have a new patch.
                    cropped_image = image[math.floor(minH):math.floor(maxH), math.floor(minW):math.floor(maxW)]
                    out_image_path = os.path.join(write_image_path, cropped_image_name)
                    cv2.imwrite(out_image_path, cropped_image)
                    # start drawing all relevant boxes on the new patch.
                    for a_box in bboxes:
                        cropped1, croppedOne = [], []                    # boxes that are completely out of this patch: ignore
                        if a_box[0]>=maxW or a_box[1]>=maxH or a_box[2]<=minW or a_box[3]<=minH:
                            continue
                        # boxes that are entirely included in patch: keep
                        elif a_box[0]>=minW and a_box[1]>=minH and a_box[2]<=maxW and a_box[3]<=maxH:
                            # print('entire')
                            cropped1 = a_box
                        # boxes that are partially included: check the extent of overlap
                        else:
                            # print('overlap')
                            areaBox = (a_box[2] - a_box[0]) * (a_box[3] - a_box[1])
                            npbox = np.array(a_box)
                            nppatch = np.array([minW, minH, maxW-1, maxH-1])
                            left_up = np.maximum(npbox[:2], nppatch[:2])
                            right_down = np.minimum(npbox[2:4], nppatch[2:4])
                            inter_section = np.maximum(right_down - left_up, 0.0)
                            inter_area = inter_section[0] * inter_section[1]
                            if inter_area >= cropTH*areaBox:
                                cropped1 = (np.concatenate([left_up, right_down])).tolist()
                                np.array(cropped1.append(a_box[4]))
                                cropped1 = np.array(cropped1)
                                # print('Successoverlap')
                            else:
                                # print('Failoverlap')
                                continue
                        # adjust coordinates
                        croppedOne[:2] = cropped1[:2] - [minW, minH]
                        croppedOne[2:4] = cropped1[2:4] - [minW, minH]
                        croppedOne.append(cropped1[4])
                        # print(minW, minH, maxW, maxH)
                        # print(a_box)
                        # print(cropped1)
                        # print(croppedOne)
                        # write a_box to new dataTxt:
                        if new_img_flag:
                            if new_line_flag:
                                f_out.write('\n')
                            new_line_flag = True
                            f_out.write(out_image_path + ' ')
                        new_img_flag = False
                        f_out.write(str(croppedOne[0]) + ',' + str(croppedOne[1]) + ',' + str(croppedOne[2]) + ',' + str(croppedOne[3]) + ',' + str(croppedOne[4]) + ' ')
                        cropped_bboxes.append(croppedOne)
                    # print(image_name, 'minW: ', minW, 'minH: ', minH, 'maxW: ', maxW, 'maxH: ', maxH, 'len(cropped_bboxes): ', len(cropped_bboxes))
                    # draw_gt_bbox(cropped_image, cropped_bboxes)
                    pass
                else: # the category is not rare: so ain't got no new patch. move on.
                    continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Offline augmentation for rare category crop",
                                     description="This script preorganizes the Taggings of Yolov3 baseline 2 format " +
                                                 "[f_data_in_path] into a desired [patch_w, patch_h] cropped Tagging " +
                                                 "file [f_data_out_path] and images [write_image_path] main dir of " +
                                                 "cropped outputs ")

    parser.add_argument('--f_data_in_path',
                        default='/home/tamar/RecceLite_code_packages/yolo3_baseline2/data/individual_datasets_without_24/recce_data_Tagging_5.txt',
                        help="the Tagging to be cropped")
    parser.add_argument('--write_image_path',
                        default='/home/tamar/DBs/Reccelite/CroppedDB/individualCropped/Cropped_1_2_3_4_5_reg_rare/',
                        help="directory path for script's image-outputs")
    parser.add_argument('--f_data_out_path',
                        default='/home/tamar/DBs/Reccelite/CroppedDB/individualCropped/5_rare.txt',
                        help="the outout of this script: the cropped Tagging")
    parser.add_argument('--patch_w', type=int, default=800, help="the width of the cropped patch")
    parser.add_argument('--patch_h', type=int, default=640, help="the height of the cropped patch")
    parser.add_argument('--cropTH', type=float, default=0.6,
                        help="above this percentage of the bbox included in the path - clip the box. " +
                             "else - ignore the bbox.")

    opts = parser.parse_args()

    main(write_image_path=opts.write_image_path, f_data_in_path=opts.f_data_in_path,
         f_data_out_path=opts.f_data_out_path, patch_w=opts.patch_w, patch_h=opts.patch_h, cropTH=opts.cropTH)
