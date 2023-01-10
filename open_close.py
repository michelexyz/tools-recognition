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
resize_and_show('original', image, 30)

# opening or closing image
open_or_close = 'close'
out = open_close(image, open_or_close, 7, 1, 2)

# Applying the Black-Hat operation
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
# blackhat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
# resize_and_show('black hat', blackhat, 30)

retval, image = cv.threshold(image, 125, 255, cv.THRESH_BINARY)

# making trackbar for number of dilation iterations
def on_dil_iter(dil_iter, img):
    er_iter = cv.getTrackbarPos('erosion iterations', 'opened-closed')
    size = cv.getTrackbarPos('kernel size', 'opened-closed')
    img = open_close(img, open_or_close, size, er_iter, dil_iter)
    resize_and_show('opened-closed', img, 30)


cv.namedWindow('opened-closed')
cv.createTrackbar('dilation iterations', 'opened-closed', 1, 10, lambda v: on_dil_iter(v, out))


# making trackbar for number of erosion iterations
def on_er_iter(er_iter, img):
    dil_iter = cv.getTrackbarPos('dilation iterations', 'opened-closed')
    size = cv.getTrackbarPos('kernel size', 'opened-closed')
    img = open_close(img, open_or_close, size, er_iter, dil_iter)
    resize_and_show('opened-closed', img, 30)


cv.createTrackbar('erosion iterations', 'opened-closed', 1, 10, lambda v: on_er_iter(v, out))


# making trackbar for eroding-dilating kernel size
def on_ker_size(size, img):
    if (size % 2) == 0:
        size += 1
    dil_iter = cv.getTrackbarPos('dilation iterations', 'opened-closed')
    er_iter = cv.getTrackbarPos('erosion iterations', 'opened-closed')
    img = open_close(img, open_or_close, size, er_iter, dil_iter)
    resize_and_show('opened-closed', img, 30)


cv.createTrackbar('kernel size', 'opened-closed', 1, 15, lambda v: on_ker_size(v, out))


cv.waitKey(0)
cv.destroyAllWindows()



