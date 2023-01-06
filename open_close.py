import cv2 as cv
import numpy as np

# receiving input a binary image, sending output a better binary image



def open_close(img, type, dim):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(dim,dim))
    if type == 'close':
        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        return closing
    else:
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return opening


image = cv.imread('C:/Users/glauc/OneDrive/Desktop/no_bg.jpg')
cv.imshow('original', image)


scale_percent = 30 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image = cv.resize(image, dim)


cv.imshow('resized', image)

out = open_close(image, 'open', 5)

#making slider

# def on_change(value):
#     print(value)
#
#
# cv.createTrackbar('slider', 'opened or closed', 0, 100, on_change)


cv.imshow('opened or closed', out)

gray = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
retval, threshed = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)
cv.imshow('binarized', threshed)

cv.waitKey(0)
cv.destroyAllWindows()



