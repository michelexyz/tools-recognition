import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

import descriptors as dsc

image = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/tools.jpg"))
hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
bg = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/green.jpg"))
print(dsc.computeColor(bg))
print("ciao")
print(dsc.computeStd(bg))
