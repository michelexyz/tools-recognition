import cv2 as cv
from useful import *


# divide the input image in sub-regions
def tassellamela(img, step, dim):

    num = 0
    width = img.shape[1]
    hight = img.shape[0]

    # calcuating taxel matrix dimensions
    rows_n = (hight-dim) // step
    columns_n = (width-dim) // step
    print('numero colonne: ' + str(columns_n) + 'numero righe: ' + str(rows_n))
    descriptors = np.zeros((rows_n + 1, columns_n + 1), dtype=np.ndarray)

    # indexes for accessing 'descriptors' matrix
    a = 0
    b = 0

    #iterating over image
    for i in range(0, hight-dim+1, step):

        for j in range(0, width-dim+1, step):

            point = (j, i)
            roi = extract_roi(img, point, dim, dim)

            # todo: applico il descrittore al tassello (ROI)
            # roi = descriptor_x(roi)

            # put the computed (descripted) roi in descriptors array
            descriptors[a][b] = roi

            resize_and_show('roi numero ' + str(num), roi, 190)
            print('tassello numero ' + str(num))

            num += 1
            b += 1

        b = 0
        a += 1


immagine = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/tools.jpg')
dim = (400, 400)
resized = cv.resize(immagine, dim, interpolation=cv.INTER_AREA)
resize_and_show('roi', resized)

step = 70
size = 100
tassellamela(resized, step, size)

cv.waitKey(0)
cv.destroyAllWindows()
