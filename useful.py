import math

import cv2 as cv
import numpy as np

from morph_fun import open_close



# NormilizeIntensity imposta ad ogni pixlel un Immagine BGR "Im"
# lo stesso valore di intensità "i_value"
# potrebbe essere anche direttamente creata una funzione "normalizeChannel"
# o ancora apply function to channel
# TODO METTI in file UTIL

def normilezeIntesity(im, i_value):
    hsvImage = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    HIm, SIm, VIm = cv.split(hsvImage)
    c, r, _ = im.shape

    VNormalized = np.full((c, r), i_value, np.uint8)
    hsvImage = cv.merge([HIm, SIm, VNormalized])
    NormImg = cv.cvtColor(hsvImage, cv.COLOR_HSV2BGR)
    return NormImg


#Questa funzione invece equalizza l'istogramma dell'intensità
def equalizeIntensity(im):
    hsvImage = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    HIm, SIm, VIm = cv.split(hsvImage)
    c, r, _ = im.shape

    # VEqualized = cv.equalizeHist(VIm)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    VEqualized = clahe.apply(VIm)
    hsvImage = cv.merge([HIm, SIm, VEqualized])
    EqImg = cv.cvtColor(hsvImage, cv.COLOR_HSV2BGR)
    return EqImg


def resize_percentage(img, scale_percent=100):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    out = cv.resize(img, dim)

    return out


#this is for showing purposes
def resize_and_show(window_name, img, scale_percent=60):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim)

    cv.imshow(window_name, resized)


# this function extract a sub image (ROI) given the coordinates
# of top left corner point and sides dimensions
def extract_roi(img, point, x_dim, y_dim):
    x, y = point
    roi = img[y:y+y_dim, x:x+x_dim]
    return roi


#Esegue la segmentazione per regioni connesse di un immagine binarizzate e restituisce un immagine ritagliata
#contente solo l'oggetto più grande
#L'immagine ritornata è booleana(0-1)
def remove_small_objects(binarized_image):
    connectivity = 4
    output = cv.connectedComponentsWithStats(binarized_image, connectivity, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output

    max_area = 0
    index_biggest_element = -1

    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(i + 1, numLabels)
            # otherwise, we are examining an actual connected component
        else:
            area = stats[i, cv.CC_STAT_AREA]
            text = "examining component {}/{}".format(i + 1, numLabels)
            #print(text)
            text = "Area : {}".format(area)
            if area > max_area:
                max_area = area #TODO finisci
                index_biggest_element = i


    area = stats[index_biggest_element, cv.CC_STAT_AREA]

    if area == 0:
        raise Exception("nessun elemento trovato")

    x = stats[index_biggest_element, cv.CC_STAT_LEFT]
    y = stats[index_biggest_element, cv.CC_STAT_TOP]
    w = stats[index_biggest_element, cv.CC_STAT_WIDTH]
    h = stats[index_biggest_element, cv.CC_STAT_HEIGHT]


    componentMaskBool = (labels[y:y + h, x:x + w] == index_biggest_element)

    return componentMaskBool.astype("uint8")

def remove_imperfections_adv(binarized_image, factor_close = 160, factor_open=160):
    area = np.count_nonzero(binarized_image)
    it_open= find_n_iterations(area, factor_open)
    it_close = find_n_iterations(area, factor_close)
    connectivity = 4

    binarized_image = open_close(binarized_image, 'open', 3, er_it=it_open, dil_it=0)

    output = cv.connectedComponentsWithStats(binarized_image, connectivity, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output

    max_area = 0
    index_biggest_element = -1

    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(i + 1, numLabels)
            # otherwise, we are examining an actual connected component
        else:
            area = stats[i, cv.CC_STAT_AREA]
            text = "examining component {}/{}".format(i + 1, numLabels)
            #print(text)
            text = "Area : {}".format(area)
            if area > max_area:
                max_area = area #TODO finisci
                index_biggest_element = i


    area = stats[index_biggest_element, cv.CC_STAT_AREA]

    if area == 0:
        raise Exception("nessun elemento trovato")

    x = stats[index_biggest_element, cv.CC_STAT_LEFT]
    y = stats[index_biggest_element, cv.CC_STAT_TOP]
    w = stats[index_biggest_element, cv.CC_STAT_WIDTH]
    h = stats[index_biggest_element, cv.CC_STAT_HEIGHT]


    componentMaskBool = (labels[y:y + h, x:x + w] == index_biggest_element).astype("uint8")

    componentMaskBool = open_close(componentMaskBool, 'open', 3, er_it=0, dil_it=it_open)
    componentMaskBool = open_close(componentMaskBool, 'close', 3, er_it=it_close, dil_it=it_close)

    return componentMaskBool

def remove_imperfections(binarized_image, factor_close = 160, factor_open=160):
    # h = binarized_image.shape[0]
    # w = binarized_image.shape[1]
    area = np.count_nonzero(binarized_image)
    it_close = find_n_iterations(area, factor_close)
    it_open = find_n_iterations(area, factor_open)


    binarized_image = open_close(binarized_image, 'open', 3, er_it=it_open, dil_it=it_open)
    binarized_image = open_close(binarized_image, 'close', 3, er_it=it_close, dil_it=it_close)
    return binarized_image



#Inserire un fattore più grandeper immagini da correggere maggiormente

def find_n_iterations(area, factor = 100):
    # area = w*h
    n_iterations = math.sqrt(area)/(factor)
    print(f'Numero di iterazioni per oggetto di area {area} e fattore {factor}: {n_iterations}')
    return round(n_iterations)


#Data la lunghezza della diagonale scelta ridimensiona l'immagine
#Mantenendo le proporzioni
def resize_to_fixed_d(im, d):
    ratio = (im.shape[0] / im.shape[1])
    # h = d * math.sqrt(ratio)
    # w = d / math.sqrt(ratio)
    #
    # dim = (int(w), int(h))
    #
    # # resize image
    # return cv.resize(im, dim,  interpolation=cv.INTER_AREA)

    fraction = d**2/(ratio**2 + 1)
    w = math.sqrt(fraction)
    h = w * ratio

    dim = (int(w), int(h))

    # resize image
    return cv.resize(im, dim,  interpolation=cv.INTER_AREA)


