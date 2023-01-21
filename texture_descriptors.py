import math
from enum import Enum

import matplotlib.pyplot as plt

# from skimage.transform import rotate
from skimage.feature import local_binary_pattern
# from skimage import data
# from skimage.color import label2rgb

from skimage import feature
import numpy as np


# import cv2 as cv


# settings for LBP
# radius = 3
# n_points = 8 * radius
#
#
# def overlay_labels(image, lbp, labels):
#     mask = np.logical_or.reduce([lbp == each for each in labels])
#     return label2rgb(mask, image=image, bg_label=0, alpha=0.5)
#
#
# def highlight_bars(bars, indexes):
#     for i in indexes:
#         bars[i].set_facecolor('r')
#
#
# # image = data.brick()
# # lbp = local_binary_pattern(image, n_points, radius, METHOD)
#
#
# def hist(ax, lbp):
#     n_bins = int(lbp.max() + 1)
#     return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
#                    facecolor='0.5')
#
#
# # plot histograms of LBP of textures
# fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
# plt.gray()
#
# titles = ('edge', 'flat', 'corner')
# w = width = radius - 1
# edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
# flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
# i_14 = n_points // 4            # 1/4th of the histogram
# i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
# corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
#                  list(range(i_34 - w, i_34 + w + 1)))
#
# label_sets = (edge_labels, flat_labels, corner_labels)
#
# for ax, labels in zip(ax_img, label_sets):
#     ax.imshow(overlay_labels(image, lbp, labels))
#
# for ax, labels, name in zip(ax_hist, label_sets, titles):
#     counts, _, bars = hist(ax, lbp)
#     highlight_bars(bars, labels)
#     ax.set_ylim(top=np.max(counts[:-1]))
#     ax.set_xlim(right=n_points + 2)
#     ax.set_title(name)
#
# ax_hist[0].set_ylabel('Percentage')
# for ax in ax_img:
#     ax.axis('off')




def compute_lbp(img, mask):
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    P = 18
    R = 70
    # dim = 2**P
    dim = (2 ** P) / P
    h_bins = np.arange(dim + 1)
    h_range = (0, dim)
    (r, c) = img.shape

    codes = local_binary_pattern(img, P, R, method="ror")
    # masked= np.ma.masked_where(mask == True, codes)
    # nmask = np.logical_not(np.logical_not(mask))
    # print(type(codes))
    # print(codes)
    # print(type(mask))
    # print(mask)
    # plt.imshow(codes)
    # plt.colorbar()
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    f, [ax0, ax1] = plt.subplots(1, 2)

    im_codes = ax0.imshow(codes)
    ax0.axis('off')
    ax0.set_title('LBP of image')
    plt.colorbar(im_codes, ax=ax0)

    x = np.ma.array(codes, mask=np.logical_not(mask))
    x.filled(fill_value=0)
    im_masked = ax1.imshow(x)
    ax1.axis('off')
    ax1.set_title('LBP of ROI')
    plt.colorbar(im_masked, ax=ax1)
    plt.show()

    h_img, _ = np.histogram(codes.ravel(), bins=h_bins, range=h_range)
    h_masked, _ = np.histogram(codes[mask], bins=h_bins, range=h_range)
    h_img = h_img / h_img.sum(dtype=np.float)
    h_masked = h_masked / h_masked.sum(dtype=np.float)

    f, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2)
    ax0.imshow(img, cmap=plt.cm.gray)
    ax0.axis('off')
    ax0.set_title('Image')
    ax1.imshow(mask, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('Mask')
    ax2.plot(h_img)
    ax2.set_title('LBP of image')
    ax3.plot(h_masked)
    ax3.set_title('LBP of ROI')
    plt.show()

UNIFORM_MULT = 0.2
UNIFORM_D = 5

ROR_MULT = 0.7
ROR_D = 4



# TODO implementa in extract_objects al posto dell'attuale compute_lbp()
class LocalBinaryPatterns:
    def __init__(self, num_points=18, radius=100, method="ror"):
        # store the number of points and radius
        self.numPoints = num_points
        self.radius = radius
        self.method = method
        self.dim = self.compute_dim(self.numPoints, self.method)

    def compute_dim(self, num_points, method):
        if method == "ror":
            return int(2 ** num_points / num_points)
        elif method == "uniform":
            return num_points + 2
        else:
            raise Exception("Invalid 'method' in LocalBinaryPatters")

    # TODO aggiungere parametro show = True per decidere se i grafici vengano mostrati o meno
    def describe(self, image, mask, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method=self.method)

        # MOSTRA LBP DELL'IMMAGINE E DELLA MASCHERA

        f, [ax0, ax1] = plt.subplots(1, 2)
        im_codes = ax0.imshow(lbp)
        ax0.axis('off')
        ax0.set_title('LBP of image')
        plt.colorbar(im_codes, ax=ax0)

        x = np.ma.array(lbp, mask=np.logical_not(mask))
        # x.filled(fill_value=0)
        im_masked = ax1.imshow(x)
        ax1.axis('off')
        ax1.set_title('LBP of ROI')
        plt.colorbar(im_masked, ax=ax1)
        plt.show()

        # CALCOLA ISTOGRAMMA DELL' LBP DELL' IMMAGINE E DELLA MASCHERA

        # (hist, _) = np.histogram(lbp.ravel(),
        #                          bins=np.arange(0, self.dim +1),
        #                          range=(0, self.dim))

        bins = np.arange(0, self.dim + 1)
        range = (0, self.dim)

        h_img, _ = np.histogram(lbp.ravel(), bins=bins, range=range)
        h_masked, _ = np.histogram(lbp[mask], bins=bins, range=range)
        h_img = h_img / (h_img.sum(dtype=np.float) + eps)
        h_masked = h_masked / (h_masked.sum(dtype=np.float) + eps)

        # # normalize the histogram
        # hist = hist.astype("float")
        # hist /= (hist.sum() + eps)

        # MOSTRA ISTOGRAMMI DI image E mask CON RELATIVE IMMAGINI

        f, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2)
        ax0.imshow(image, cmap=plt.cm.gray)
        ax0.axis('off')
        ax0.set_title('Image')
        ax1.imshow(mask, cmap=plt.cm.gray)
        ax1.axis('off')
        ax1.set_title('Mask')
        ax2.plot(h_img)
        ax2.set_title('LBP of image')
        ax3.plot(h_masked)
        ax3.set_title('LBP of ROI')
        plt.show()

        # return the histogram of Local Binary Patterns
        return h_masked

class find_r_mode(Enum):
    AREA = "ar"
    WIDTH = "wi"

r_mode= find_r_mode.AREA

def parametric_lbp(num_points=18, method="ror",width = 500, area = 250000):#TODO 0
    global UNIFORM_MULT
    global UNIFORM_D
    global ROR_MULT
    global ROR_D
    global r_mode

    if method == "ror":
        radius = find_r(width=width, area=area, mode = r_mode, mult=ROR_MULT, d= ROR_D)
        return LocalBinaryPatterns(num_points=num_points, radius=radius, method=method)
    elif method == "uniform":
        radius = find_r(width=width, area=area, mode=r_mode, mult=UNIFORM_MULT, d=UNIFORM_D)
        return LocalBinaryPatterns(num_points=num_points, radius=radius, method=method)

def find_r(width = 500, area = 250000, mode=find_r_mode.WIDTH, mult = 0.2, d = 5):#TODO tune mult and d
    if find_r_mode.AREA == mode:
        return math.sqrt(area / math.pi) * mult
    if find_r_mode.WIDTH == mode:
        return width/d
