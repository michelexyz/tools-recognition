import cv2 as cv
import numpy as np


def resize_percentage(img, scale_percent=100):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    out = cv.resize(img, dim)

    return out


#this is for showing purposes
def resize_and_show(window_name, img, scale_percent=60):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim)

    cv.imshow(window_name, resized)


# this function extract a sub image (ROI) given the coordinates
# of top left corner point and sides dimensions
def extract_roi(img, point, x_dim, y_dim):
    x, y = point
    roi = img[y:y+y_dim, x:x+x_dim]
    return roi
