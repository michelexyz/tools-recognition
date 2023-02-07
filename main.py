import os
import sys

import cv2 as cv

from descriptors.shape_descriptors import CallableHu
from detect_objects import detect_objects

import numpy as np

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


# Leggi immagine con oggetti multipli
image = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/tools.jpg"))

# Leggi immagine del relativo background
bg = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/green.jpg"))

# CARICA IL MODELLO
data = np.load('cl1.npy', allow_pickle=True)

m = data[0]

# CREA OGGETTO DESCRITTORE
descriptor = CallableHu(draw=True)

detect_objects(image,bg,obj_classifier=m, descriptor=descriptor)
# cv.waitKey(0)
#
# cv.destroyAllWindows()
#
# cv.waitKey(1)

sys.exit(0)
