import numpy as np
import cv2 as cv
from enum import Enum
import matplotlib.pyplot as plt


# class color_stats:
#   rgb: int = None
#   hsv: int = None
#   ycbcr: int = None
#
#   def __init__(self, rgb, hsv, ycbcr):
#


#
# class compute_color:
#   rgb: bool = True
#   hsv: bool = True
#   ycbcr: bool = False
#   texel: np.ndarray = None
#   return: np.ndarra
#
#   class type(Enum):
#     AVARAGE = 0
#     DOMINANT = 1
#
#   def __init__(self, texel, rgb, hsv, ycbcr, t):
#     if (t == type.AVARAGE):
#
#
#     elif:
#
#   def myfunc(abc):
#     print("Hello my name is " + abc.name)

class stats_type(Enum):
    AVARAGE = "av"
    DOMINANT = "do"


def computeColor(texel: np.ndarray, code=stats_type.AVARAGE, rgb=True, hsv=True, ycbcr=True):
    n_spaces = 0
    # Potrei direttamente aggiungere la possibilitò di passare le funzioni di conversioni come argomento
    # e quindi avere a disposizione un infinità di spazi colore
    if rgb:
        n_spaces += 1
    if hsv:
        n_spaces += 1
    if ycbcr:
        n_spaces += 1

    if n_spaces == 0:
        # print("Nussuno spazio colore selezionato")
        raise Exception("Nussuno spazio colore selezionato")

    if texel.ndim == 3:
        _, _, ch = texel.shape
    elif texel.ndim == 2:
        ch = 1
    else:
        raise Exception("numero dimensioni del texel uguale a %d " % texel.ndim)
    print(ch)
    out = np.zeros((n_spaces, ch))

    i = 0

    # TODO aggiungi slider con pesi spazio colore
    if code == stats_type.AVARAGE:
        # dtype= np.uint8 sbagliato
        # np.empty per maggiore efficienza

        if (rgb):
            avg = avarageColor(texel)
            out[i] = avg
            i += 1
            #stampa il colore medio
            avg_patch = np.ones(shape=texel.shape, dtype=np.uint8) * np.uint8(avg)

            # plt.imshow(avg_patch)
            # plt.axis('off')
            # plt.show()
        # TODO vedere se aggiungere spazio colore LAB(invece di hsv che è ciclico)
        if (hsv):
            hsvImage = cv.cvtColor(texel, cv.COLOR_BGR2HSV)
            out[i] = avarageColor(hsvImage)
            i += 1

        if (ycbcr):
            ycbcrImage = cv.cvtColor(texel, cv.COLOR_BGR2YCR_CB)
            out[i] = avarageColor(ycbcrImage)
            i += 1

    # TODO aggiungi slider numero colori
    elif code == stats_type.DOMINANT:

        if (rgb):
            out[i] = dominantColors(texel)
            i += 1

        if (hsv):
            hsvImage = cv.cvtColor(texel, cv.COLOR_BGR2HSV)
            out[i] = dominantColors(hsvImage)
            i += 1

        if (ycbcr):
            ycbcrImage = cv.cvtColor(texel, cv.COLOR_BGR2YCR_CB)
            out[i] = dominantColors(ycbcrImage)
            i += 1

    return out


def computeStd(texel: np.ndarray, rgb=True, hsv=True, ycbcr=True):
    n_spaces = 0
    # Potrei direttamente aggiungere la possibilitò di passare le funzioni di conversioni come argomento
    # e quindi avere a disposizione un infinità di spazi colore
    if rgb:
        n_spaces += 1
    if hsv:
        n_spaces += 1
    if ycbcr:
        n_spaces += 1

    if n_spaces == 0:
        # print("Nussuno spazio colore selezionato")
        raise Exception("Nussuno spazio colore selezionato")

    _,_,ch = texel.shape
    print(ch)
    out = np.zeros((n_spaces, ch))

    i = 0
    if rgb:
        out[i] = std(texel)
        i += 1
    if hsv:
        hsvImage = cv.cvtColor(texel, cv.COLOR_BGR2HSV)
        out[i] = std(hsvImage)
        i += 1
    if ycbcr:
        ycbcrImage = cv.cvtColor(texel, cv.COLOR_BGR2YCR_CB)
        out[i] = std(ycbcrImage)
        i += 1
    return out

def avarageColor(texel: np.ndarray):
    # pixels = texel.reshape(-1, 3)
    return np.mean(texel, axis=0).mean(axis=0)
    # if not (rgb | hsv | ycbcr):
    #     print("Nussuno spazio colore selezionato")
    # if(rgb):
    #
    #
    #
    #
    # if(hsb):
    #
    #
    # if(ycbcr):


def dominantColors(texel: np.ndarray, n_colors=5):
    pixels = np.float32(texel.reshape(-1, 3))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]


    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices] / float(counts.sum())]))
    rows = np.int_(texel.shape[0] * freqs)

    dom_patch = np.zeros(shape=texel.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    plt.imshow(dom_patch)
    plt.axis('off')
    plt.show()

    return palette


def std(texel: np.ndarray):
    pixels = texel.reshape(-1, 3)
    return np.std(pixels, axis=0)
