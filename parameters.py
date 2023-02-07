import cv2 as cv

#LBP parameters
method = "ror"# TODO tune
P = 17 # TODO tune

UNIFORM_MULT = 0.2
UNIFORM_D = 5

ROR_MULT = 0.8
ROR_D = 4


#Connected components segmentation
connectivity = 4 #TODO tune

#Morphological functions
kShape = cv.MORPH_CROSS # TODO tune
