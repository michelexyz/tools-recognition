import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import numpy as np
from tr_utils.useful import resize_and_show
from tesselation import tassellamela

PATCH_SIZE = 45

# image = cv.imread('C:/Users/glauc/PycharmProjects/tools-recognition/data/non_omogeneus_backgrounds/blue_square/IMG20230120125942.jpg')
image = cv.imread('/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/tools.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# showing read image
resize_and_show('image', image, 30)

# Full image
distances = [3, 5, 7]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# graycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
# GLCM = graycomatrix(gray, distances, angles)
# energy = graycoprops(GLCM, 'energy')[0, 0]

# print('energy is ' + str(energy))

#
# cell_locations = [(50, 50), (1000, 50), (50, 350), (200, 350)]
# cell_patches = []
# for loc in cell_locations:
#     cell_patches.append(gray[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
#
# # select some patches from sky areas of the image
# scratch_locations = [(50, 200), (150, 200), (250, 150), (200, 200), (2000, 2000)]
# scratch_patches = []
# for loc in scratch_locations:
#     scratch_patches.append(gray[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

tassellata = tassellamela(gray, 50, 50)

# compute some GLCM properties for each patch
diss_sim = []
corr = []
homogen = []
energy = []
contrast = []
num = 0
for row in tassellata:
    for tassello in row:
        # glcm = graycomatrix(tassello, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        glcm = graycomatrix(tassello, distances, angles, levels=256, symmetric=True, normed=True)
        diss_sim.append(graycoprops(glcm, 'dissimilarity')[0, 0]) #[0,0] to convert array to value
        corr.append(graycoprops(glcm, 'correlation')[0, 0])
        homogen.append(graycoprops(glcm, 'homogeneity')[0, 0])
        energy.append(graycoprops(glcm, 'energy')[0, 0])
        contrast.append(graycoprops(glcm, 'contrast')[0, 0])
        print('gray co props computed for tassello num: '+str(num))
        num += 1


# OPTIONAL PLOTTING for Visualization of points and patches
# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
# ax.imshow(gray, cmap=plt.cm.gray, vmin=0, vmax=255)
# for (y, x) in cell_locations:
#     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
#
# for (y, x) in scratch_locations:
#     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')

# ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)

ax.plot(diss_sim[:], corr[:], 'go', label='all')

# ax.plot(diss_sim[:len(cell_patches)], corr[:len(cell_patches)], 'go', label='cells')
# ax.plot(diss_sim[len(cell_patches):], corr[len(cell_patches):], 'bo', label='Scratch')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
# for i, patch in enumerate(cell_patches):
#     ax = fig.add_subplot(3, len(cell_patches), len(cell_patches)*1 + i + 1)
#     ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
#     ax.set_xlabel('Cells %d' % (i + 1))

# for i, patch in enumerate(scratch_patches):
#     ax = fig.add_subplot(3, len(scratch_patches), len(scratch_patches)*2 + i + 1)
#     ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
#     ax.set_xlabel('Scratch %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
#daje forzaroma

cv.waitKey(0)
cv.destroyAllWindows()
