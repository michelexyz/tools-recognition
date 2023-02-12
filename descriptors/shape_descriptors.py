import math

import cv2 as cv

from math import copysign, log10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib import pyplot as plt

from descriptors import DescriptorInterface


class CallableHu(DescriptorInterface):
    """Extract text from a PDF."""

    def __init__(self, dim=7):
        #self.draw = draw
        self.dim = dim

    def describe(self, componentMask, componentMaskBool, area, name, draw= True):
        """Overrides DescriptorInterface.describe()"""

        return hu_moments(componentMask, draw=draw,  name=name)


    def get_dim(self):
        """Overrides DescriptorInterface.get_dim()"""
        return self.dim


def hu_fun(componentMask, componentMaskBool, area):
    return hu_moments(componentMask, draw=True)


def hu_moments(im, draw=False,name = 'Default name'):
    # Calculate Moments
    moments = cv.moments(im)
    # Calculate Hu Moments
    huMoments = cv.HuMoments(moments).reshape(-1)

    for i in range(0, 7):
        # Log scale hu moments
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))

    if draw:
        # MOSTRA Immagine e relativi valori di hu moments
        f, [ax0, ax1] = plt.subplots(1, 2, figsize=(10,3), gridspec_kw={'width_ratios': [1, 2]}, constrained_layout = True)
        f.suptitle(name)

        ax0.imshow(im, cmap=plt.cm.gray)
        ax0.axis('off')
        ax0.set_title('Image')
        index = []
        for k, _ in enumerate(huMoments):
            index.append(f'hu[{k}]')

        cells = np.empty((len(index), 2), dtype=object)
        cells[:,0] = index
        cells[:,1] = huMoments

        #mpl.rcParams.update({'font.size': 22})

        ax1.table(cellText=cells, colLabels=['index', 'values'], loc='center', colWidths = [1/4,1/2], cellLoc ='center')
        ax1.axis('off')
        ax1.set_title('Hu moments')
        # set the spacing between subplots

        plt.show()





    return huMoments


# TODO aggiungi colonne di diverso colore
def plot_moments(images, names):
    # # make this example reproducible
    # np.random.seed(0)
    #
    # # define figure and axes
    # fig, ax = plt.subplots()
    #
    # # hide the axes
    # fig.patch.set_visible(False)
    # ax.axis('off')
    # ax.axis('tight')
    #
    # # create data
    # df = pd.DataFrame(np.random.randn(20, 2), columns=['First', 'Second'])
    # df.iloc[1][1] = cv.imread('/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/tools.jpg')
    #
    # # create table
    # table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    #
    # # display table
    # fig.tight_layout()
    # plt.show()

    # first, we'll create a new figure and axis object
    fig, ax = plt.subplots(figsize=(10, 12))

    # set the number of rows and cols for our table
    rows = images.shape[0]

    # space for images and names
    data_offset = 3
    cols = 7 + data_offset

    # create a coordinate system based on the number of rows/columns

    # adding a bit of padding on bottom (-1), top (1), right (0.5), left (0.5)
    y_pad = 1
    ax.set_ylim(-y_pad, rows + y_pad)

    x_pad = 0.5
    ax.set_xlim(-x_pad, cols + x_pad)

    # sets the font size for the table
    font_size = 12

    # TITLE
    ax.set_title(
        'Hu moments',
        loc='left',
        fontsize=18,
        weight='bold'
    )

    # REMOVE AXIS
    ax.axis('off')

    # DRAW LINES
    for row in range(rows):
        ax.plot(
            [-x_pad, cols + x_pad],
            [row - .5, row - .5],
            ls=':',
            lw='.5',
            c='grey'
        )
    ax.plot([-x_pad, cols + x_pad], [rows - 0.5, rows - 0.5], lw='.5', c='black')

    # DRAW DATA ON EVERY ROW
    for row in range(rows):

        h = hu_moments(images[row])

        # draw the name of the image
        ax.text(x=0, y=row, s=names[row], va='center', ha='left', fontsize=font_size)

        # draw the hu moments values
        for i, val in enumerate(h):
            if i == 0:
                ax.text(x=i + data_offset, y=row, s="%.3f" % val, va='center', ha='left', fontsize=font_size)
            else:

                ax.text(x=i + data_offset, y=row, s="%.3f" % val, va='center', ha='left', fontsize=font_size)

    # DRAW COLUMNS NAMES
    headers_y = rows
    for i in range(cols):
        if i == 0:
            ax.text(x=i, y=headers_y, s="Names", va='center', ha='left', weight='bold', fontsize=font_size)
        elif i == 1:
            ax.text(x=i, y=headers_y, s="Images", va='center', ha='left', weight='bold', fontsize=font_size)
        elif i >= (data_offset):
            ax.text(x=i, y=rows, s=f"hu[{i - data_offset}]", va='center', ha='left', weight='bold', fontsize=font_size)

    # DRAW IMAGES #TODO draw image to aspecif location?
    # adds new axis to the figure
    newaxes = []
    for row in range(rows):
        # offset each new axes by a set amount depending on the row

        # this is probably the most fiddly aspect (TODO: some neater way to automate this)
        newaxes.append(
            fig.add_axes([.22, .725 - (row * .064), .12, .06])
        )

    for i, ax in enumerate(newaxes):
        ax.imshow(images[i], cmap=plt.cm.gray)
        ax.axis('off')

        # ax.text(x=.5, y=row, s=h, va='center', ha='left')
        # # shots column - this is my "main" column, hence bold text
        #
        # #ax.text(x=2, y=row, s=h, va='center', ha='right', weight='bold')
        # # passes column
        #
        # ax.text(x=3, y=row, s=d['passes'], va='center', ha='right')
        # # goals column
        #
        # ax.text(x=4, y=row, s=d['goals'], va='center', ha='right')
        # # assists column
        #
        # ax.text(x=5, y=row, s=d['assists'], va='center', ha='right')
    plt.show()

if __name__ == '__main__':
    images = np.empty(10, dtype=object)
    names = np.empty(10, dtype=object)

    for i in range(10):
        images[i] = cv.imread('/Users/michelevannucci/PycharmProjects/ToolsRecognition/no_bg.jpg', cv.IMREAD_GRAYSCALE)
        names[i] = 'test'
    plot_moments(images, names)
    hu_moments(images[0], draw=True)
