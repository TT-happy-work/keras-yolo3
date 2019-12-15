import os
from xml.etree import ElementTree
from scipy.spatial import distance as dist
import numpy as np
from time import sleep
import cv2
from PIL import Image, ImageDraw
import colorsys

def order_points(ptsArr):

    # pt_a, pt_b: out of the 2 left X,  a is thr top one and b is the bottom.
    # pt_c, pt_d: out of the 2 right X, c is the

    ## sort the points based on their x-coordinates
    xSorted = ptsArr[np.argsort(ptsArr[:, 0]), :]

    ## grab the 2 left-most and the 2 right-most points from the sorted
    ## x-coodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    ## now, sort the left-most coordinates according to their
    ## y-coordinates so we can grab the top-left and bottom-left
    ## points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (pt_a, pt_b) = leftMost

    ## now that we have the top-left coordinate, use it as an
    ## anchor to calculate the Euclidean distance between the
    ## top-left and right-most points; by the Pythagorean
    ## theorem, the point with the largest distance will be
    ## our bottom-right point
    # D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    # (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # the remaing two points are arranged
    # so that pt_c is the lower of the two
    rightMost = rightMost[np.argsort(rightMost[:, 1])[::-1], :]
    (pt_c, pt_d) = rightMost

    ## return the coordinates in top-left, top-right,
    ## bottom-right, and bottom-left order
    return np.array([pt_a, pt_b, pt_c, pt_d], dtype="float32")


if __name__ == '__main__':
    ## XML folder to convert to data text
    mainDbFolder = '/home/tamar/DBs/Reccelite/All_data/Tagging3'
    ## class list file
    labelsFile = '/home/tamar/DBs/Reccelite/All_data/class_names.txt' # extract labels dict from recce_names file
    ## Write contents of dictionary allImgs into recce_dataset.txt as required by k-means
    fout = '/home/tamar/DBs/Reccelite/All_data/dataTxt_4points_Tagging3.txt'
    ## file to write annomalities into
    annomalitiesFile = '/home/tamar/DBs/Reccelite/All_data/Annomalities_3.txt'

    num_pts_per_tar = 4 #how many points per target to write to datatext

    debug_prints = False

    subFolderList = os.listdir(mainDbFolder)
    allImgs = []; # list of Img-dictionaries. each dict is the complete info of an img.
    annomal = []; # keep track of all sorts of problems
    with open(labelsFile, 'r') as f:
        labels = {}
        for i, tar in enumerate(f):
            labels[tar[:-1]] = i
    for singleImg in subFolderList: # new Img
        first_line_flag = True
        singleImgAnnomalities = {}
        annotationFile = os.path.join(mainDbFolder, singleImg, singleImg) + '.xml'  # the name of the xml-file is similar to the name of the sub-folder
        for file in os.listdir(os.path.join(mainDbFolder, singleImg)): # many of the xmls have shorter names
            if file.endswith(".xml"):
                os.rename(os.path.join(mainDbFolder, singleImg, file), annotationFile)
        if os.path.isfile(os.path.join(mainDbFolder, singleImg, singleImg) + '.jpg'):
            imgFilePath = os.path.join(mainDbFolder, singleImg, singleImg) + '.jpg'
        elif os.path.isfile(os.path.join(mainDbFolder, singleImg, singleImg) + '.png'):
            imgFilePath = os.path.join(mainDbFolder, singleImg, singleImg) + '.png'
        else:
            Exception('no image in dir')
        if singleImg == 'XMLs' or not os.path.isfile(annotationFile):
            singleImgAnnomalities['Img'] = imgFilePath
            singleImgAnnomalities['Problem'] = 'No Annotations'
            singleImgAnnomalities['tarID'] = 'ALL'
            annomal.append(singleImgAnnomalities)
            singleImgAnnomalities = {}
            continue
        singleImage_dict = {}
        singleImage_dict['imagePath'] = imgFilePath
        singleImage_dict['allTargetsInImg'] = []
        allTargets = {}
        f = open(annotationFile,'rt')
        tree = ElementTree.parse(f)
        root = tree.getroot()
        for item in root.findall("./WorldPoints/WorldPoint"): # goes through each to collect all target-types with their ordinals
            singleImgAnnomalities = {}
            a = item.find('ID').text
            b = item.find('Name').text
            if debug_prints: print('type: ', a, b)
            if not b: # invalid cls Type
                singleImgAnnomalities['Img'] = imgFilePath
                singleImgAnnomalities['tarID'] = a
                singleImgAnnomalities['Problem'] = 'No Name Annotation'
                annomal.append(singleImgAnnomalities)
                singleImgAnnomalities = {}
                continue
            if b not in labels.keys():  # if this is a tarType not encountered before, add it to dict and to the labels_txt
                if debug_prints: print('tarTYPE: ', b, 'Image:', imgFilePath)
                labels[b] = max(labels.values())+1
                with open(labelsFile, 'a+') as f:
                        f.write(b)
                        f.write('\n')
            allTargets[item.find('ID').text] = item.find('Name').text  # keys=IDs; values=tar-type
        for item in root.findall("./Appearances/MarkedImage/SensorPointWorldPointPairs/SensorPointWorldPointPair"): # goes through each to collect all ordinal target types
            pts = [];
            if item.find("./First/Shape").text == 'PointPolygon':
                coo_count=0;
                for coo in item.findall("./First/Coordinate"):
                    if not coo_count: # first coo of pointpolygon is not relevant.
                        coo_count = 1
                        if debug_prints: print('before pop', tar_id)
                        if item.findall("./Second/WorldPointId")[0].text == tar_id:  # if belongs to the Polygon that was before - eliminate the polygon.
                            singleImage_dict['allTargetsInImg'].pop()  # get rid of the polygon the was before this pointpolygon, if same ID
                        continue
                    [pts.append(x.text) for x in coo.findall("./X")]
                    [pts.append(y.text) for y in coo.findall("./Y")]
                    coo_count+=1
                if debug_prints: print('\nPointPolygon: ', tar_id, pts)
            elif item.find("./First/Shape").text != 'Polygon':  # Protection from non-polygon entries + keep track of problem
            # if item.find("./First/Shape").text != 'Polygon':  # Protection from non-polygon entries + keep track of problem
                singleImgAnnomalities['Img'] = imgFilePath
                singleImgAnnomalities['tarID'] = item.findall("./Second/WorldPointId")[0].text
                singleImgAnnomalities['Problem'] = item.find("./First/Shape").text
                annomal.append(singleImgAnnomalities)
                singleImgAnnomalities = {}
                continue
            else:
                # try:
                #     if item.find("./First/partially_hidden").text:
                #         continue
                # except:
                #     pass
                tar_id = item.findall("./Second/WorldPointId")[0].text
                for coo in item.findall("./First/Coordinate"):
                    [pts.append(x.text) for x in coo.findall("./X")]
                    [pts.append(y.text) for y in coo.findall("./Y")]
                if debug_prints: print('\nPolygon: ', tar_id, pts)
            ptsArr = np.reshape(np.asarray(pts, dtype=np.float32), (int(len(pts)/2), 2))
            # tl = np.array((np.min(ptsArr[:, 0]), np.min(ptsArr[:, 1])))
            # br = np.array((np.max(ptsArr[:, 0]), np.max(ptsArr[:, 1])))
            if ptsArr.shape[0] !=4:
                if debug_prints: print(annotationFile, item.findall("./Second/WorldPointId")[0].text)
            pt_a, pt_b, pt_c, pt_d = order_points(ptsArr)  # in order to obtain x_min etc correctly
            tar = {}
            if item.findall("./Second/WorldPointId")[0].text not in allTargets.keys():
                continue
            tar['tarID'] = int(item.findall("./Second/WorldPointId")[0].text)         # holds the ordinal index of tar
            tar['tarClass'] = allTargets[item.findall("./Second/WorldPointId")[0].text]
            tar['tarX_a'] = pt_a[0]
            tar['tarY_a'] = pt_a[1]
            tar['tarX_b'] = pt_b[0]
            tar['tarY_b'] = pt_b[1]
            tar['tarX_c'] = pt_c[0]
            tar['tarY_c'] = pt_c[1]
            tar['tarX_d'] = pt_d[0]
            tar['tarY_d'] = pt_d[1]
            singleImage_dict['allTargetsInImg'].append(tar)
        allImgs.append(singleImage_dict)
    f = open(fout, "w")
    for imgIdx in range(len(allImgs)):
        if first_line_flag:
            f.write(allImgs[imgIdx]['imagePath'] + ' ') # write to file the image Name, after which will foloow all info of all tars in img.
        else:
            f.write('\n' + allImgs[imgIdx]['imagePath'] + ' ') # write to file the image Name, after which will foloow all info of all tars in img.
        first_line_flag = False
        img = Image.open(allImgs[imgIdx]['imagePath'])
        draw = ImageDraw.Draw(img)
        ## draw settings
        hsv_tuples = [(x / len(labels), 0.9, 1.0) for x in range(len(labels))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        numOfTarsInImg = len(allImgs[imgIdx]['allTargetsInImg'])
        imgShape = [cv2.imread(allImgs[imgIdx]['imagePath']).shape[0], cv2.imread(allImgs[imgIdx]['imagePath']).shape[1]]
        showImg = []
        for tar in range(numOfTarsInImg):
            ## DBG: To show only requested targets
            #a = ['unknow', 'jeepprivate']
            #if not any(x in allImgs[imgIdx]['allTargetsInImg'][tar]['tarClass'] for x in a):
                #continue
            # showImg.append(1)
            ## Write to data txt file
            if allImgs[imgIdx]['allTargetsInImg'][tar]['tarClass'] == 'unknow':
                continue
            # f.write('#' + str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarID']) + '#')
            f.write(str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarX_a']) + ',' + str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarY_a']) + ',') #  write the a coordinates
            f.write(str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarX_b']) + ',' + str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarY_b']) + ',') #  write the a coordinates
            f.write(str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarX_c']) + ',' + str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarY_c']) + ',') #  write the a coordinates
            f.write(str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarX_d']) + ',' + str(allImgs[imgIdx]['allTargetsInImg'][tar]['tarY_d']) + ',') #  write the a coordinates
            f.write(str(labels[allImgs[imgIdx]['allTargetsInImg'][tar]['tarClass']]) + ' ')
            ## Show the boxes on image
            # bbox = np.array((allImgs[imgIdx]['allTargetsInImg'][tar]['tarX_a'], allImgs[imgIdx]['allTargetsInImg'][tar]['tarY_a'], allImgs[imgIdx]['allTargetsInImg'][tar]['tarX_b'], allImgs[imgIdx]['allTargetsInImg'][tar]['tarY_b'], allImgs[imgIdx]['allTargetsInImg'][tar]['tarX_c'], allImgs[imgIdx]['allTargetsInImg'][tar]['tarY_c'], allImgs[imgIdx]['allTargetsInImg'][tar]['tarX_d'], allImgs[imgIdx]['allTargetsInImg'][tar]['tarY_d']))
            # tarText = allImgs[imgIdx]['allTargetsInImg'][tar]['tarClass']
            ## DBG: to search only for requested targets
            #if not any(x in tarText for x in a):
                #continue
            # bbox_text = "%s" % tarText
            # text_size = draw.textsize(bbox_text)
            # bbox_reshaped = list(bbox.reshape(4, 2).reshape(-1))
            # draw.rectangle(bbox_reshaped, outline=colors[labels[allImgs[imgIdx]['allTargetsInImg'][tar]['tarClass']]], width=3)
            # text_origin = bbox_reshaped[:2] - np.array([0, text_size[1]])
            # draw.rectangle([tuple(text_origin), tuple(text_origin + text_size)], fill=colors[labels[allImgs[imgIdx]['allTargetsInImg'][tar]['tarClass']]])
            ## draw bbox
            # draw.text(tuple(text_origin), bbox_text, fill=(0, 0, 0))
            ## DBG: to show only images containing requested targets
            # img.show()
            # sleep(2)
            # img.close()
    f.close()
    ## Record in file the collected annomalities
    annoF = open(annomalitiesFile, "w")
    for ann in annomal:
        annoF.write('\n ----------------------- \nImage Path:' + ann['Img'] + '; \nThe Problem: ' + ann['Problem'] + '; \nthe Target ID: ' + ann['tarID'])
    annoF.close()