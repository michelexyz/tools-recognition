import math
from enum import Enum

import matplotlib.pyplot as plt

# from skimage.transform import rotate
from skimage.feature import local_binary_pattern
# from skimage import data
# from skimage.color import label2rgb

from skimage import feature
import numpy as np

from descriptors import DescriptorInterface
from tr_utils.parameters import *


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
    """

    :param img:
    :param mask:
    """
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


# TODO implementa in extract_objects al posto dell'attuale compute_lbp()
class LocalBinaryPatterns:
    def __init__(self, num_points=18, radius=100, method="ror"):
        # store the number of points and radius
        self.numPoints = num_points
        self.radius = radius
        self.method = method
        self.dim = self.compute_dim(self.numPoints, self.method)

    def compute_dim(self, num_points, method):# TODO togli parametri gia presenti nell'oggetto
        """

        Parameters
        ----------
        num_points
        method

        Returns
        -------

        """
        if method == "ror":
            return self.ror_dim(num_points)
            # return 2 ** num_points
        elif method == "uniform":
            return num_points + 2
        else:
            raise Exception("Invalid 'method' in LocalBinaryPatters")

    # TODO Forse inutile
    def ror_dim(self, num_points):
        """Gray scale and rotation invariant LBP (Local Binary Patterns).
            LBP is an invariant descriptor that can be used for texture classification.
            Parameters
            ----------
            image : (N, M) array
                Graylevel image.
            P : int
                Number of circularly symmetric neighbour set points (quantization of
                the angular space).
            R : float
                Radius of circle (spatial resolution of the operator).
            method : {'default', 'ror', 'uniform', 'var'}
                Method to determine the pattern.
                * 'default': original local binary pattern which is gray scale but not
                    rotation invariant.
                * 'ror': extension of default implementation which is gray scale and
                    rotation invariant.
                * 'uniform': improved rotation invariance with uniform patterns and
                    finer quantization of the angular space which is gray scale and
                    rotation invariant.
                * 'nri_uniform': non rotation-invariant uniform patterns variant
                    which is only gray scale invariant [2]_, [3]_.
                * 'var': rotation invariant variance measures of the contrast of local
                    image texture which is rotation but not gray scale invariant.
            Returns
            -------
            output : (N, M) array
                LBP image.
            References
            ----------
            .. [1] T. Ojala, M. Pietikainen, T. Maenpaa, "Multiresolution gray-scale
                   and rotation invariant texture classification with local binary
                   patterns", IEEE Transactions on Pattern Analysis and Machine
                   Intelligence, vol. 24, no. 7, pp. 971-987, July 2002
                   :DOI:`10.1109/TPAMI.2002.1017623`"""

        total_combinations = 2 ** num_points
        half_num = int(num_points / 2)
        repeating_patterns_addends = []
        repeating_patterns_sizes = []
        repeating_patterns_sizes.append(half_num)

        for i in np.flip(np.arange(1, half_num)):

            for s in repeating_patterns_sizes:
                if s % i > 0:
                    repeating_patterns_sizes.append(i)

        for r in repeating_patterns_sizes:  # TODO optimize
            addend = (2 ** r) * r
            repeating_patterns_addends.append(addend)
        repeating_patterns_num = np.sum(repeating_patterns_addends)

        distinct_patterns_num = int((total_combinations - repeating_patterns_num) / num_points) + repeating_patterns_num

        print(f"distinct patterns: {distinct_patterns_num}")

        return distinct_patterns_num

    # TODO aggiungere parametro show = True per decidere se i grafici vengano mostrati o meno
    def describe(self, image, mask, name='Default name', draw=True):
        eps = 1e-7  # TODO

        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method=self.method)

        # MOSTRA LBP DELL'IMMAGINE E DELLA MASCHERA
        if draw == True:
            f, [ax0, ax1] = plt.subplots(1, 2)
            f.suptitle(name)

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
        if draw == True:
            f, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2)
            f.suptitle(name)

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


r_mode = find_r_mode.AREA


# Given the method of the LBP and the area of the object finds a proper radius and creates an lbp
# Object with it(scale indipendent)


def parametric_lbp(num_points=18, method="ror", width=500, area=250000):  # TODO 0
    global UNIFORM_MULT
    global UNIFORM_D
    global ROR_MULT
    global ROR_D
    global r_mode

    if method == "ror":
        radius = find_r(width=width, area=area, mode=r_mode, mult=ROR_MULT, d=ROR_D)

    elif method == "uniform":
        radius = find_r(width=width, area=area, mode=r_mode, mult=UNIFORM_MULT, d=UNIFORM_D)

    print("Radius: {}".format(radius))

    return LocalBinaryPatterns(num_points=num_points, radius=radius, method=method)


# Funzione passabile come parametro "descrittore" a detect_objects() e describe_data()

def lbp_fun(componentMask, componentMaskBool, area):
    lbpDescriptor = parametric_lbp(P, method=method, area=area)
    lbp = lbpDescriptor.describe(componentMask, componentMaskBool)
    return lbp


class CallableLbp(DescriptorInterface):
    """Extract text from an email."""

    def __init__(self, P, method):
        self.P = P
        self.method = method
        self.dim = LocalBinaryPatterns().compute_dim(self.P, self.method)

    def describe(self, componentMask, componentMaskBool, area, name, draw=True):
        """Overrides DescriptorInterface.describe()"""

        lbpDescriptor = parametric_lbp(self.P, method=self.method, area=area)
        lbp = lbpDescriptor.describe(componentMask, componentMaskBool, name=name, draw=draw)
        return lbp

    def get_dim(self):
        """Overrides DescriptorInterface.get_dim()"""

        return self.dim


def find_r(width=500, area=250000, mode=find_r_mode.WIDTH, mult=0.2, d=5):  # TODO tune mult and d
    if find_r_mode.AREA == mode:
        return math.sqrt(area / math.pi) * mult
    if find_r_mode.WIDTH == mode:
        return width / d
