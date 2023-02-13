import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# performs k-means classifier on input data
def k_means(data, k, criteria):

    # converting data to float32 type
    data = np.float32(data)

    # performing k-means
    ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now separate the data, Note the ravel()
    A = data[label.ravel() == 0]
    B = data[label.ravel() == 1]

    # Plot the data
    # plt.scatter(A[:, 0], A[:, 1])
    # plt.scatter(B[:, 0], B[:, 1], c='r')
    # plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    # plt.xlabel('feature1'), plt.ylabel('feature2')
    # plt.show()

    return A, B, label



# questo è un codice di esempio, su un input di 50 dati descritti da 2 features
# X = np.random.randint(25,50,(25,2))
# Y = np.random.randint(60,85,(25,2))
# H = np.random.randint(10,14,(20,2))
#
# # dati su cui fare k-means
# Z = np.vstack((X,Y,H))
#
# # convert to np.float32
# Z = np.float32(Z)
#
# # define criteria and apply kmeans()
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret,label,center = cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
#
# # Now separate the data, Note the flatten()
# A = Z[label.ravel()==0]
# B = Z[label.ravel()==1]
#
# # Plot the data
# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1],c = 'r')
# plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()



#todo: - dai un input sensato a k-means
#      - prova prima con 2 features, poi con +
#      - capisci come plottare i risultati
#      - fai quella roba della ground truth
