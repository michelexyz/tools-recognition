import os

import numpy as np
import cv2 as cv
from useful import *

def kmeans_quantization(img, classes):

    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = classes
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    quantized = res.reshape((img.shape))

    return quantized

# image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/some_tools_dalmata_bg.jpg')
#
#
# quant = kmeans_quantization(image, 4)
# resize_and_show('quantizzata', quant, 40)
#
# # path = 'C:/Users/glauc/OneDrive/Desktop/UNI/elaborazione immagini digitali/output/quantized'
# # cv.imwrite(os.path.join(path, 'quant4.jpg'), quant)
#
#
# cv.waitKey(0)
# cv.destroyAllWindows()
