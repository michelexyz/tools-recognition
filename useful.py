import cv2 as cv
import numpy as np


def resize_percentage(img, scale_percent=100):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    out = cv.resize(img, dim)

    return out
