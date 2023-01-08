import cv2 as cv
import numpy as np
from useful import *

# receiving input a binary image, sending output a better binary image



def open_close(img, type, dim, er_it=1, dil_it=1):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(dim,dim))
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


# image = cv.imread('C:/Users/glauc/OneDrive/Desktop/UNI/elaborazione immagini digitali/progetto/images/no_bg.jpg')
image = cv.imread('C:/Users/glauc/OneDrive/Desktop/UNI/elaborazione immagini digitali/progetto/images/no_bg_gamma.jpg')
cv.imshow('original', image)


retval, image = cv.threshold(image, 125, 255, cv.THRESH_BINARY)


cv.imshow('resized', image)

out = open_close(image, 'close', 5, 1, 2)

#making slider

# def on_change(value):
#     print(value)
#
#
# cv.createTrackbar('slider', 'opened or closed', 0, 100, on_change)

# resizing for displaying purposes
out = resize_percentage(out, 30)


cv.imshow('opened or closed', out)

cv.waitKey(0)
cv.destroyAllWindows()



