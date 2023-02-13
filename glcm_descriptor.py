import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from skimage import io
import cv2 as cv
import numpy as np
from useful import resize_and_show
from tesselation import tassellamela
import os
from PIL import Image
from kmeans_classifier import k_means


# returns an array formatted for k-means with K = 2
def k2_means_data_format(feature1, feature2):
    formatted = [[feature1],[feature2]]
    return formatted

# returns an array formatted for k-means with K = 3
def k3_means_data_format(feature1, feature2, feature3):
    formatted = [[feature1],[feature2],[feature3]]
    return formatted

# returns an array formatted for k-means with K = 4
def k4_means_data_format(feature1, feature2, feature3, feature4):
    formatted = [[feature1],[feature2],[feature3],[feature4]]
    return formatted


# draws rectangles of 2 different colors depending which class
# the tassello belongs to
def kmeans2_out(labels, tassellata, img, step, patch_dim):

    # getting img dimensions
    width = img.shape[1]
    hight = img.shape[0]

    # iterating on image step by step
    lab_i = 0
    for i in range(0, hight-patch_dim+1, step):
        lab_j = 0
        for j in range(0, width-patch_dim+1, step):

            # getting top left and bottom right coordinates
            top_left = (i,j)
            bottom_right = (i+patch_dim,j+patch_dim)

            # draw a rectangle
            if labels[lab_i][lab_j] == 0:
                # this area is labelled "0", so
                # draw a RED rectangle
                cv.rectangle(img, top_left, bottom_right, (0,0,255), 3)
            else:
                # this area is labelled "1", so
                # draw a BLUE rectangle
                cv.rectangle(img, top_left, bottom_right, (255,0,0), 3)

            # increment labels index "lab_j" (for column)
            lab_j += 1

        # increment labels index "lab_i" (for rows)
        lab_i += 1


    return 0


def k3_out(labels, tassellata, img, step, patch_dim):
    return 0

# TODO rendere tutto una funzione chiamabile
# def glcm_descriptor(image, PATCH_SIZE, STEP):

PATCH_SIZE = 30
STEP = PATCH_SIZE

# image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/blue_square/IMG20230120125942.jpg')
image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/pinza.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# showing read image
resize_and_show('image', image, 30)

# distances and angles con i quali calcolo GLCM matrix
# distances = [3, 5, 7]
distances = [5]
# angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
angles = [0]

# splitting the image into tasselli
tassellata = tassellamela(gray, STEP, PATCH_SIZE)

# get tassellata size, to initialize GLCM properties array's size
tassellated_size = tassellata.shape[0] * tassellata.shape[1]

# initializing some GLCM properties arrays
diss_sim = np.empty(tassellated_size, dtype=np.ndarray)
corr = np.empty(tassellated_size, dtype=np.ndarray)
homogen = np.empty(tassellated_size, dtype=np.ndarray)
energy = np.empty(tassellated_size, dtype=np.ndarray)
contrast = np.empty(tassellated_size, dtype=np.ndarray)
num = 0

# calculating some properties for each tassello's GLCM
for row in tassellata:
    for tassello in row:
        # glcm = graycomatrix(tassello, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        glcm = graycomatrix(tassello, distances, angles, levels=256, symmetric=True, normed=True)
        # diss_sim.append(graycoprops(glcm, 'dissimilarity')[0, 0]) #[0,0] to convert array to value
        diss_sim[num] = graycoprops(glcm, 'dissimilarity')[0, 0]
        # corr.append(graycoprops(glcm, 'correlation')[0, 0])
        corr[num] = (graycoprops(glcm, 'correlation')[0, 0])
        # homogen.append(graycoprops(glcm, 'homogeneity')[0, 0])
        homogen[num] = (graycoprops(glcm, 'homogeneity')[0, 0])
        # energy.append(graycoprops(glcm, 'energy')[0, 0])
        energy[num] = (graycoprops(glcm, 'energy')[0, 0])
        # contrast.append(graycoprops(glcm, 'contrast')[0, 0])
        contrast[num] = (graycoprops(glcm, 'contrast')[0, 0])
        print('gray co props computed for tassello num: ' + str(num))

        num += 1

# now i have all the descripted taxels
# i need to classify them

# formatting the data, using dissimilarity and correlation
features_kmeans = k2_means_data_format(diss_sim, corr)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# use K-MEANS with K = ?
K = 2
# this returns the objects of the first and second classes separated, and the "labels" array is a 2 dimensions array
# saying "the object with index n belongs to the class labels[n]" (credo)

class1, class2, labels = k_means(features_kmeans, K, criteria)

# reshaping the labels array
labels2 = labels.reshape(tassellata.shape)

# showing on the image which tassello belongs to which class
# depending on how many classes we choose

if K == 2:
    kmeans2_out(labels2, tassellata, image, STEP, PATCH_SIZE)
elif K == 3:
    k3_out(labels2, tassellata, image, STEP, PATCH_SIZE)





#
# # OPTIONAL PLOTTING for Visualization of points and patches
# # create the figure
# fig = plt.figure(figsize=(12, 12))
#
# # display original image with locations of patches
# ax = fig.add_subplot(3, 2, 1)
# # ax.imshow(gray, cmap=plt.cm.gray, vmin=0, vmax=255)
# # for (y, x) in cell_locations:
# #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
# #
# # for (y, x) in scratch_locations:
# #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
#
# # ax.set_xlabel('Original Image')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.axis('image')
#
# # for each patch, plot (dissimilarity, correlation)
# ax = fig.add_subplot(3, 2, 2)
#
# ax.plot(diss_sim[:], corr[:], 'go', label='all')
#
# # ax.plot(diss_sim[:len(cell_patches)], corr[:len(cell_patches)], 'go', label='cells')
# # ax.plot(diss_sim[len(cell_patches):], corr[len(cell_patches):], 'bo', label='Scratch')
# ax.set_xlabel('GLCM Dissimilarity')
# ax.set_ylabel('GLCM Correlation')
# ax.legend()
#
# # display the image patches
# # for i, patch in enumerate(cell_patches):
# #     ax = fig.add_subplot(3, len(cell_patches), len(cell_patches)*1 + i + 1)
# #     ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
# #     ax.set_xlabel('Cells %d' % (i + 1))
#
# # for i, patch in enumerate(scratch_patches):
# #     ax = fig.add_subplot(3, len(scratch_patches), len(scratch_patches)*2 + i + 1)
# #     ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
# #     ax.set_xlabel('Scratch %d' % (i + 1))
#
#
# # display the patches and plot
# fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
# plt.tight_layout()
# plt.show()
# #daje forzaroma

cv.waitKey(0)
cv.destroyAllWindows()
