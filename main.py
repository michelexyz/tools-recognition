import os
import sys

import cv2 as cv

from analize_data import analize_data
from descriptors import DescriptorCombiner
from descriptors.shape_descriptors import CallableHu
from descriptors.texture_descriptors import CallableLbp
from detect_objects import detect_objects

import numpy as np

from prepare_data import describe_data

import parameters
from train import prepare_models

# import cv2 as cv
# import os
# from matplotlib import pyplot as plt
# import numpy as np
#
# image = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/tools.jpg"))
# hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# H = hsvImage[:, :, 0]
# S = hsvImage[:, :, 1]
# V = hsvImage[:, :, 2]
# #h, s, v = cv.split(hsvImage)
# plt.imshow(S)
# plt.xticks([]), plt.yticks([])
# plt.show()
# # display = 0
#
# # display.convertTo(S, 255, 1, 0)
# # cv.applyColorMap( display, display, 2)
# # cv.imshow("imagesc",display)
#
#
# cv.imshow('Original image', image)
# cv.imshow('HSV image', hsvImage)
# # cv.imshow('Saturation', S)
#
# th2 = cv.adaptiveThreshold(S, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
#                            cv.THRESH_BINARY, 49, 0)
#
# plt.imshow(th2, 'gray')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
# kernelOpen = cv.getStructuringElement(cv.MORPH_RECT, [3, 3])
# kernelClose = cv.getStructuringElement(cv.MORPH_RECT, [5, 5])
# opening = cv.morphologyEx(th2, cv.MORPH_OPEN, kernelOpen)
# closing = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernelClose)
#
# # eroded = cv.erode(th2.copy(), None, iterations=)
# cv.imshow("eroded", closing)
#
# # FLOOD
#
# # im_floodfill = closing.copy()
# # h, w = closing.shape[:2]
# # mask = np.zeros((h+2, w+2), np.uint8)
# # cv.floodFill(im_floodfill, mask, (100,100), (255,255,255))
# #cv.imshow("filled", im_floodfill)
#
# output = cv.connectedComponentsWithStats(closing, 4, cv.CV_32S)
# (numLabels, labels, stats, centroids) = output
#
# imgBoxed = closing.copy()
#
# for i in range(0, numLabels):
# 	# if this is the first component then we examine the
# 	# *background* (typically we would just ignore this
# 	# component in our loop)
# 	if i == 0:
# 		text = "examining component {}/{} (background)".format(
# 			i + 1, numLabels)
# 	# otherwise, we are examining an actual connected component
# 	else:
# 		text = "examining component {}/{}".format( i + 1, numLabels)
# 	# print a status message update for the current connected
# 	# component
# 	print("[INFO] {}".format(text))
# 	# extract the connected component statistics and centroid for
# 	# the current label
# 	x = stats[i, cv.CC_STAT_LEFT]
# 	y = stats[i, cv.CC_STAT_TOP]
# 	w = stats[i, cv.CC_STAT_WIDTH]
# 	h = stats[i, cv.CC_STAT_HEIGHT]
# 	area = stats[i, cv.CC_STAT_AREA]
#
# 	print("Area: {}".format(area))
# 	(cX, cY) = centroids[i]
# 	if area > 100:
# 		cv.rectangle(imgBoxed, (x, y), (x + w, y + h), (255, 255, 255), 3)
# 		cv.circle(imgBoxed, (int(cX), int(cY)), 4, (0, 0, 255), -1)
#
# cv.imshow("imgBoxed", imgBoxed)

#TODO gestione eccezioni dei file
# TODO UI E analisi dei file specifici

print('yo')
# CREA I DESCRITTORI
hu = CallableHu()
lbp = CallableLbp(P=parameters.P, method=parameters.method)
print('yo')
# COMBINA I DESCRITTORI
descriptor = DescriptorCombiner([hu, lbp])



describe_str = 'describe'
analize_str = 'analyze'
train_str = 'train'
detect_str = 'detect'

default_pipe = [describe_str, analize_str, train_str, detect_str]

pipe_line = [describe_str]

pipe_line = default_pipe

#pipe_line = [detect_str]

#feature_extracted = False

f_extraction_data = None

with_extracted_data = True

dataset_path="/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/processed"

im_path = os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/tools.jpg")

bg_path = os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/green.jpg")

op_num = 1

if describe_str in pipe_line:
    print(f"Operazione {op_num}: DESCRIZIONE DEI DATI")
    op_num += 1

    #DESCRIVE LE IMMAGINI E SALVA I DATI SU FILE
    describe_data(descriptor, dataset_path=dataset_path,output_file='data.npy')



if analize_str in pipe_line:
    print(f"Operazione {op_num}: ANALISI DEI DATI E FEATURE EXTRACION")
    op_num += 1



    scaler, mds, optimal_n = analize_data('data.npy')
    f_extraction_data = scaler, mds, optimal_n

    #feature_extracted = True

if train_str in pipe_line:
    print(f"Operazione {op_num}: TRAINING DEL MODELLO")
    op_num += 1

    #ALLENA I MODELLI E LI SALVA SU FILE

    if with_extracted_data:
        prepare_models('data_extracted.npy', 'cl1_extracted.npy')

    else:
        prepare_models('data_extracted.npy', 'cl1.npy')


if detect_str in pipe_line:
    print(f"Operazione {op_num}: OBJECT DETECTION")
    op_num += 1
    # Leggi immagine con oggetti multipli
    image = cv.imread(im_path)

    # Leggi immagine del relativo background
    bg = cv.imread(bg_path)


    # CARICA IL MODELLO
    m = None
    if with_extracted_data:
        cl_data = np.load('cl1_extracted.npy', allow_pickle=True)

        m = cl_data[0]


        # CARICA I DATI DI FEATURE EXTRACTION
        data = np.load('extraction_data.npy', allow_pickle=True)
        f_extraction_data = data[0]

    else:
        cl_data = np.load('cl1.npy', allow_pickle=True)
        m = cl_data[0]







    detect_objects(image,bg,obj_classifier=m, descriptor=descriptor, f_extraction_data=f_extraction_data)
# cv.waitKey(0)
#
# cv.destroyAllWindows()
#
# cv.waitKey(1)

sys.exit(0)
