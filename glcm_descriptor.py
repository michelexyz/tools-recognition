import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import numpy as np




# returns an array formatted for k-means with K = 2
def k2_means_data_format(feature1, feature2):
    formatted = np.column_stack((feature1, feature2))
    return formatted

# returns an array formatted for k-means with 3 features
def k3_means_data_format(feature1, feature2, feature3):
    formatted = np.column_stack((feature1, feature2, feature3))
    return formatted

# returns an array formatted for k-means with 4 features
def k4_means_data_format(feature1, feature2, feature3, feature4):
    formatted = np.column_stack((feature1, feature2, feature3, feature4))
    return formatted

# returns an array formatted for k-means with 5 features
def k5_means_data_format(feature1, feature2, feature3, feature4, feature5):
    formatted = np.column_stack((feature1, feature2, feature3, feature4, feature5))
    return formatted



# draws rectangles of 2 different colors depending which class
# the tassello belongs to
def kmeans2_out(labels, img, step, patch_dim):

    # getting img dimensions
    width = img.shape[1]
    hight = img.shape[0]

    # iterating on image step by step
    # and on label "array" 1 by 1
    lab_i = 0
    for i in range(0, hight-patch_dim+1, step):

        for j in range(0, width-patch_dim+1, step):

            # getting top left and bottom right coordinates
            top_left = (j,i)
            bottom_right = (j+patch_dim,i+patch_dim)

            # draw a rectangle
            if labels[lab_i] == 0:
                # this area is labelled "0", so
                # draw a RED rectangle
                cv.rectangle(img, top_left, bottom_right, (0,0,255), 3)
            else:
                # this area is labelled "1", so
                # draw a BLUE rectangle
                cv.rectangle(img, top_left, bottom_right, (255,0,0), 3)

            # increment labels index "lab_i"
            lab_i += 1


    return img


def k3_out(labels, tassellata, img, step, patch_dim):
    return 0

def glcm_descriptor(image):

    # distances and angles con i quali calcolo GLCM matrix
    # distances = [3, 5, 7]
    distances = [5]
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    angles = [0]

    # computing GLCM of current tassello
    glcm = graycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)

    # calculating some properties for each tassello's GLCM
    diss_sim = (graycoprops(glcm, 'dissimilarity')[0, 0])
    corr = (graycoprops(glcm, 'correlation')[0, 0])
    homogen = (graycoprops(glcm, 'homogeneity')[0, 0])
    energy = (graycoprops(glcm, 'energy')[0, 0])
    contrast = (graycoprops(glcm, 'contrast')[0, 0])

    # print('gray co props computed for tassello num: ' + str(num))

    features = [diss_sim, corr, homogen, energy, contrast]

    return np.array(features)
