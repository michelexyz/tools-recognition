import cv2 as cv

#LBP parameters
method = "ror"# TODO tune
P = 18 # TODO tune

#Connected components segmentation
connectivity = 4 #TODO tune

#Morphological functions
kShape = cv.MORPH_CROSS # TODO tune