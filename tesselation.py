import cv2 as cv
from useful import *


# divide the input image in sub-regions
def tassellamela(img, step, dim, descriptor_funct=None):

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

            # applico il descrittore al tassello (ROI)
            # but only if the function is not None
            if descriptor_funct is not None:
                descripted_roi = descriptor_funct(roi, num)
            else:
                descripted_roi = roi

            # put the computed (descripted) roi in descriptors matrix
            descriptors[a][b] = descripted_roi

            # resize_and_show('roi numero ' + str(num), roi, 190)
            print('tassello numero ' + str(num))

            num += 1
            b += 1

        b = 0
        a += 1
    return descriptors


# example function, writes a number on an image
def numera(img, numero):
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (img.shape[0]//2,img.shape[1]//2)
    new_img = cv.putText(img, str(numero), org, font, 1, (255,0,0), 1, cv.LINE_AA)
    return new_img

# immagine = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/tools.jpg')
# dim = (400, 400)
# resized = cv.resize(immagine, dim, interpolation=cv.INTER_AREA)
# resize_and_show('roi', resized)
#
# step = 40
# size = 50
#
# # function to be passed in 'tassellamela'
# # this funct, will be applied to all rois
# funct = numera
#
# tassellamela(resized, step, size, funct)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
