import cv2 as cv
from useful import *


# divide the input image in sub-regions
def tassellamela(img, step, dim):

    num = 0

    #iterating over image
    for i in range(0, img.shape[0]-dim+1, step):
        for j in range(0, img.shape[1]-dim+1, step):
            point = (j, i)
            roi = extract_roi(img, point, dim, dim)

            resize_and_show('roi numero ' + str(num), roi, 200)

            num += 1


immagine = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/tools.jpg')
dim = (400, 400)
resized = cv.resize(immagine, dim, interpolation=cv.INTER_AREA)
resize_and_show('roi', resized)

step = 50
dim = 200
tassellamela(resized, step, dim)

cv.waitKey(0)
cv.destroyAllWindows()
