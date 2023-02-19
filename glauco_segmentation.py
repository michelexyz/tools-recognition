# questo è tipo il main dove segmentare le immagini
import os

import cv2 as cv
import numpy as np

import glcm_descriptor
from quantization import kmeans_quantization
from tesselation import tassella_e_descrivi
from tr_utils.useful import resize_and_show
from kmeans_classifier import k_means
from quantization import kmeans_quantization

# READ AN IMAGE
# image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/blue_square/IMG20230120125942.jpg')
# image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/pinza.jpg')

# dalmata_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/dalmata/sfondo_dalmata_1.jpg')

# blue_square_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/blue_square/blue_square_1.jpg')

# red_square_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/red_square/red_square_1.jpg')

# caffettiere_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/square_caffettiere/caffettiere_1.jpg')

# swag_pattern_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/swag_pattern/swag_1.jpg')

# worm_1
image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/worms/worm_1.jpg')


print('immagine letta')
# showing read image
resize_and_show('image', image, 20)

PATCH_SIZE = 30
STEP = 20

# number of features that called describer will extract (GLCM = 5)
num_features = 2

# Quantizated image k = 10 seems to work well with GLCM descriptor
image = kmeans_quantization(image, 10)
resize_and_show('quantizzata', image, 20)
print('immagine quantizzata')
# cv.waitKey(0)

# GRAYSCALE converting
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print('immagine grayscale')

# SPLITTING the image into tasselli AND
# DESCRIBEING each tassello
tassellata_descritta = tassella_e_descrivi(gray, STEP, PATCH_SIZE, num_features, glcm_descriptor.glcm_descriptor)
print('immagine TASSELLATA E DESCRITTA')


save_description = cv.waitKey(0)
# save description to file if 'a' is pressed

if save_description == ord('a'):
    save_data_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/descripted_data/'
    output_file = f'worm_1_quant10_glcm_patch{PATCH_SIZE}_step{STEP}_corr_diss.npy'
    np.save(save_data_path + output_file, tassellata_descritta)
    print('descrizione salvata nel file')

# CLASSIFICATION based on prev. description

# formatting the data for k-means, using dissimilarity and correlation
# features_kmeans = k2_means_data_format(diss_sim, corr)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# use K-MEANS with K classes
K = 2
# this returns the objects of the first and second classes separated, and the "labels" array is a 2 dimensions array
# saying "the object with index n belongs to the class labels[n]" (credo)
class1, class2, labels = k_means(tassellata_descritta, K, criteria)

# reshaping the labels array
# labels2 = labels.reshape(tassellata.shape)

# showing on the image which tassello belongs to which class
# depending on how many classes we choose.
#
if K == 2:
    out = glcm_descriptor.kmeans2_out(labels, image, STEP, PATCH_SIZE)
elif K == 3:
    out = glcm_descriptor.k3_out(labels, image, STEP, PATCH_SIZE)

# showing output
resize_and_show('segmentata', out, 20)

# key = cv.waitKey(0)
#
# if key == ord('s'):
#     path = 'C:/Users/glauc/OneDrive/Desktop/UNI/elaborazione immagini digitali/output/'
#     cv.imwrite(os.path.join(path, 'glcm_kmeans_2features_pach30_step20.jpg'), out)
#     print('salvata immagine')

cv.waitKey(0)
cv.destroyAllWindows()


