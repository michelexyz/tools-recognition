from math import copysign, log10

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from descriptors import DescriptorInterface


class CallableHu(DescriptorInterface):
    """Extract text from a PDF."""

    def __init__(self, dim=7):
        #self.draw = draw
        self.dim = dim

    def describe(self, componentMask, componentMaskBool=None, name='Default Hu name', draw= True):
        """Overrides DescriptorInterface.describe()"""

        return hu_moments(componentMask, draw=draw, name=name)


    def get_dim(self):
        """Overrides DescriptorInterface.get_dim()"""
        return self.dim, [self.dim]


    # Descriptions can be specified in case we want to pass a mean description for example
    def draw_tabular(self, samples, means = None, stds = None):
        n_samples = samples.shape[0]
        masks = np.empty(n_samples, dtype=object)
        categories = np.empty(n_samples, dtype=object)
        for i, sample in enumerate(samples):
            (binarized, binarized_bool, masked, category) = sample
            masks[i] = binarized
            categories[i] = category
        if not(means is None ):
            plot_moments(masks, categories,title='media', moments=means)

        if not(stds is None):
            plot_moments(masks, categories,title='deviazione std', moments=stds)
        elif means is None:#Default case where both means and sts are None
            plot_moments(masks, categories, moments=None)
    def draw_samples(self, samples):
        n_samples = samples.shape[0]
        for i, sample in enumerate(samples):
            (binarized, binarized_bool, masked, category) = sample
            self.describe(binarized, name = category, draw = True)



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
def plot_moments(images, names, title = '', moments = None):

    # first, we'll create a new figure and axis object
    fig, ax = plt.subplots(figsize=(12, int(images.shape[0]*1.3)))

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
        f'Hu moments {title}',
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
        if moments is None:
            h = hu_moments(images[row])
        else:
            h = moments[row]

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
            fig.add_axes([.24, .735 - (row * (.658/rows)), 1.2/rows, .6/rows])
        )
    # Flips to meach data in rows
    images = np.flip(images)
    for i, ax in enumerate(newaxes):
        ax.imshow(images[i], cmap=plt.cm.gray)
        ax.axis('off')


    plt.show()

if __name__ == '__main__':
    images = np.empty(10, dtype=object)
    names = np.empty(10, dtype=object)

    for i in range(10):
        # images[i] = cv.imread('/Users/michelevannucci/PycharmProjects/ToolsRecognition/no_bg.jpg', cv.IMREAD_GRAYSCALE) TODO replace path
        names[i] = 'test'
    plot_moments(images, names)
    hu_moments(images[0], draw=True)
