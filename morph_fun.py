import cv2 as cv
import numpy as np
from parameters import kShape
#FUNZIONI MORFOLOGICHE #TODO rimetti nel file originale da modificare con if (__name__ == '__main__'):

def open_close(img, type, dim = 1, er_it=1, dil_it=1, shape = kShape):
    kernel = cv.getStructuringElement(shape,(int(dim),int(dim)))
    if type == 'close':
        # closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel) # dilation -> erosion
        dilated = cv.dilate(img, kernel, iterations=dil_it)
        closing = cv.erode(dilated, kernel, iterations=er_it)
        return closing
    else:
        # opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel) # erosion -> dilation
        eroded = cv.erode(img, kernel, iterations=er_it)
        opening = cv.dilate(eroded, kernel, iterations=dil_it)
        return opening