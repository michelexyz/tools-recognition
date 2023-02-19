# questo è tipo il main dove segmentare le immagini
# però prende un file che contiene già la descrizione dell' immagine letta.
# quindi salta il passaggio della descrizione
import os

import cv2 as cv
import numpy as np

import glcm_descriptor
from quantization import kmeans_quantization
from tesselation import tassella_e_descrivi
from tr_utils.useful import resize_and_show
from kmeans_classifier import k_means
from quantization import kmeans_quantization

# READ AN IMAGE and corrispondent DESCRIPTION
# image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/blue_square/IMG20230120125942.jpg')
# image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/pinza.jpg')
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/dalmata/sfondo_dalmata_1.jpg')

# dalmata_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/dalmata/sfondo_dalmata_1.jpg')
#descriprion_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/descripted_data/sfondo_dalmata_1_glcm_patch30_step20_corr_diss.npy'

# blue_square_1
image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/blue_square/blue_square_1.jpg')
descriprion_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/descripted_data/blue_square_1_quant10_glcm_patch30_step20_corr_diss.npy'

# red_square_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/red_square/red_square_1.jpg')
#descriprion_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/descripted_data/red_square_1_quant10_glcm_patch30_step20_corr_diss.npy'

# caffettiere_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/square_caffettiere/caffettiere_1.jpg')
#descriprion_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/descripted_data/caffettiere_1_quant10_glcm_patch30_step20_corr_diss.npy'

# swag_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/swag_pattern/swag_1.jpg')
#descriprion_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/descripted_data/swag_1_quant10_glcm_patch30_step20_corr_diss.npy'

# worm_1
#image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/worms/worm_1.jpg')
#descriprion_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/descripted_data/worm_1_quant10_glcm_patch30_step20_corr_diss.npy'

print('immagine letta')
# showing read image
resize_and_show('image', image, 20)

descripted_img = np.load(descriprion_path, allow_pickle=True)
print('descrizione letta')


# these are needed for showing classification results
# devono essere gli stessi valori con cui è stata descritta è immagine
PATCH_SIZE = 30
STEP = 20



# CLASSIFICATION based on read. description

# formatting the data for k-means, using dissimilarity and correlation
# features_kmeans = k2_means_data_format(diss_sim, corr)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# use K-MEANS with K classes
K = 2
# this returns the objects of the first and second classes separated, and the "labels" array is a 2 dimensions array
# saying "the object with index n belongs to the class labels[n]" (credo)
class1, class2, labels = k_means(descripted_img, K, criteria)
print('classificazione fatta')

# reshaping the labels array
# labels2 = labels.reshape(tassellata.shape)

# showing on the image which tassello belongs to which class
# depending on how many classes we choose.
#
if K == 2:
    out = glcm_descriptor.kmeans2_out(labels, image, STEP, PATCH_SIZE)
    # resize_and_show('rettengoly??', out, 10.3)
    print('tasselli colorati disegnati')
elif K == 3:
    out = glcm_descriptor.k3_out(labels, image, STEP, PATCH_SIZE)

# showing output
resize_and_show('segmentata (out)', out, 20)

key = cv.waitKey(0)

# per prendere il nome del file letto avendo il path:
filename = os.path.basename(descriprion_path)
filename = os.path.splitext(filename)
filename = filename[0]


# per salvare l' immagine premi 's'
if key == ord('s'):
    path = 'C:/Users/glauc/OneDrive/Desktop/UNI/elaborazione immagini digitali/output/'
    cv.imwrite(os.path.join(path, filename +'.jpg'), out)
    print('immagine salvata')

cv.destroyAllWindows()


