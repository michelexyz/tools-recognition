import math

import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

import descriptors as dsc

import texture_descriptors as tx

from texture_descriptors import LocalBinaryPatterns

from morph_fun import open_close

from models import Models

import pandas as pd
from IPython.display import display

# NormilizeIntensity imposta ad ogni pixlel un Immagine BGR "Im"
# lo stesso valore di intensità "i_value"
# potrebbe essere anche direttamente creata una funzione "normalizeChannel"
# o ancora appyl fun to channel
# TODO METTI in file UTIL

def normilezeIntesity(im, i_value):
    hsvImage = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    HIm, SIm, VIm = cv.split(hsvImage)
    c, r, _ = im.shape

    VNormalized = np.full((c, r), i_value, np.uint8)
    hsvImage = cv.merge([HIm, SIm, VNormalized])
    NormImg = cv.cvtColor(hsvImage, cv.COLOR_HSV2BGR)
    return NormImg


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


# Gamma correction can be implemented using lookup table (LUT). It maps the input pixel values to the output values.
# For each pixel value in the range [0, 255] is calculated a corresponding gamma corrected value. OpenCV provides the
# LUT function which performs a lookup table transform.
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    res = cv.LUT(src, table)
    # cv.imshow("test", res)
    # cv.waitKey(0)
    return res


# Trova la gamma data un'intesità media obbiettivo
def optimalGamma(goal_perc, mean):
    if goal_perc <= 0 or goal_perc >= 1:
        raise Exception("goal_mean fuori dal range (0,1) valore %d" % goal_perc)

    exp = math.log(goal_perc) / math.log(mean / 255)
    gamma = 1 / exp
    print(f"goal: {goal_perc}, mean: {mean}, gamma: {gamma}")
    return gamma


# Applica l'operazione di gamma sul canale di insità per eliminare le ombre
def gammaIntensity(im, gamma):
    # cv.imshow("vediamo se copy to funziona ori", im)
    hsv_image = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    HIm, SIm, VIm = cv.split(hsv_image)

    VGamma = gammaCorrection(VIm, gamma)
    hsv_image = cv.merge([HIm, SIm, VGamma])
    # cv.imshow("vediamo se copy to funziona gam", cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR))
    # np.copyto(dst=im, src=cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR))
    return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)


def on_gamma(g, im, normIm):
    # cv.imshow("vediamo se on gamma to funziona ori", NormIm)
    np.copyto(normIm, gammaIntensity(im, g / 10))

    cv.imshow("gamma", normIm)


def on_mean(m_perc, mean):
    goal_m = m_perc / 100
    optimal_g = optimalGamma(goal_m, mean)
    cv.setTrackbarPos('gamma', 'gamma', int(optimal_g * 10))


# Leggi immagine con oggetti multipli
image = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/tools.jpg"))
cv.imshow("original image", image)
# gammaIntensity(image, 2)
# cv.imshow("test 10", image)
cv.waitKey(0)
# Leggi immagine del relativo background
bg = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/green.jpg"))
cv.imshow("original bg", bg)

# Estrai valore di intensità medio dalla stessa
# hsvBg = cv.cvtColor(bg, cv.COLOR_BGR2HSV)
# _, _, VMeanBg = dsc.avarageColor(hsvBg)

# clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))

# Estrai il valore di intesità medio dall' immagine da analizzare
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
_, _, V = cv.split(hsv)
v_mean = np.mean(V)

print("vmean: %d" % v_mean)

# NormIm = normilezeIntesity(image, VMeanBg)
# NormIm = equalizeIntensity(image)
# Immagine normalizzata che sarà aggiornata dalle operazioni successive
NormIm = np.copy(image)

# cv.imshow('NormIm', NormIm)

# FINESTRA per l'alterazione della gamma
cv.namedWindow("gamma")
trackbar_k = 'gamma'
# imposta le trackbar per la gamma e l'intensità media obbiettivo
cv.createTrackbar(trackbar_k, "gamma", 1, 100, lambda v: on_gamma(v, image, normIm=NormIm))
cv.createTrackbar("media", "gamma", 1, 100, lambda v: on_mean(v, v_mean))

# Inizializza la l'intensità media obiettivo al 50%
cv.setTrackbarPos('media', 'gamma', 80)

# Attendi la selezione della gamma da parte dell'utente per passare alla fase successiva
# cv.waitKey(0) #TODO

# Memorizza l'intensità media obiettivo fornita dall'utente
goalPerc = cv.getTrackbarPos('media', 'gamma') / 100

# NormBg = normilezeIntesity(bg, VMeanBg)
# NormBg = equalizeIntensity(bg)
# NormBg = gammaIntensity(bg, 2)

# Estrai il valore di intesità medio dal bg
hsv = cv.cvtColor(bg, cv.COLOR_BGR2HSV)
_, _, V = cv.split(hsv)
bgVMean = np.mean(V)

# Normalizza l'mmagine di bg con la media obiettivo fornita dall'utente
op = optimalGamma(goal_perc=goalPerc, mean=bgVMean)
print("GAMMA OTTIMALE %f" % op)

NormBg = gammaIntensity(bg, op)
cv.imshow("normilized bg", NormBg)

# Mostra anche l'immagine da analizzare normalizzata
cv.imshow("normilize image", NormIm)

# calcola i valori BGR medi dell'immagine di bg
bgRgbMean = dsc.computeColor(NormBg)[0]
# BMean = bgRgbMean[0]
# GMean = bgRgbMean[1]
# RMean = bgRgbMean[2]
print(bgRgbMean)

# calcola la dev standard dell'immagine di bg
bgRgbStd = dsc.computeStd(NormBg)[0]
std = dsc.computeStd(image)[0]
print(std)
# BStd = bgRgbStd[0]
# GStd = bgRgbStd[1]
# RStd = bgRgbStd[2]
print(bgRgbStd)


# Binarizza l'immagine data la media dei valori e la deviazione std di un'immagine di bg
def extract_objects(im, k, channel_mean: np.ndarray, std: np.ndarray):
    if channel_mean.shape != std.shape:
        raise Exception("dimensione dell'array della media di canali diversa da quello della std")
    lower_bound = channel_mean - (std * k)
    upper_bound = channel_mean + (std * k)
    bg_threshold = cv.inRange(im, lower_bound, upper_bound)
    obj_threshold = cv.bitwise_not(bg_threshold)
    return obj_threshold


# Al movimento dello slider estrai gli oggeti con un nuovo k (val)
def on_trackbar(val):
    # k = val
    # # TODO trasforma in una funzione il codice sotto
    # lowBg = (BMean - BStd * k, GMean - GStd * k, RMean - RStd * k)
    # highBg = (BMean + BStd * k, GMean + GStd * k, RMean + RStd * k)
    # bg_threshold = cv.inRange(NormIm, lowBg, highBg)
    # obj_threshold = cv.bitwise_not(bg_threshold)
    # cv.imshow(title_window, obj_threshold)
    # cv.imwrite("no_bg.jpg", obj_threshold)
    obj_thresh = extract_objects(NormIm, val, bgRgbMean, bgRgbStd)
    cv.imshow(bin_window, obj_thresh)


# Crea una finestra dove viene mostrata la binarizazzione dell'immagine dato il k selezionato con lo slider
alpha_slider_max = 10
bin_window = 'Immagine con slider'
cv.namedWindow(bin_window)
trackbar_k = 'K slider %d' % alpha_slider_max
cv.createTrackbar(trackbar_k, bin_window, 0, alpha_slider_max, on_trackbar)
# Show some stuff
cv.setTrackbarPos(trackbar_k, bin_window, 6)

# Una volta premuto un tasto salva l'immagine
# cv.waitKey(0) #TODO
k = cv.getTrackbarPos(trackbar_k, bin_window)

obj_thresh = extract_objects(NormIm, k, bgRgbMean, bgRgbStd)
cv.imwrite('no_bg_gamma.jpg', obj_thresh)

kShape = cv.MORPH_CROSS
erosionIt = 2  # TODO tune
# La dimensione del kernel deve essere = 3 per il funzionamento del programma(vedi dentro if area..)
obj_thresh = open_close(obj_thresh, 'open', 3, er_it=erosionIt, dil_it=0, shape=kShape)

connectivity = 4
output = cv.connectedComponentsWithStats(obj_thresh, connectivity, cv.CV_32S)
(numLabels, labels, stats, centroids) = output



# clone our original image (so we can draw on it) and then draw
# a bounding box surrounding the connected component along with
# a circle corresponding to the centroid
rectImage = image.copy()
r, c, _ = image.shape

#CARICA IL MODELLO
data = np.load('cl1.npy', allow_pickle=True)

m = data[0]



# loop over the number of unique connected component labels
# TODO erode prima di estrazione dei singoli componenti per separare le "macche"
# TODO E fare in modo che vengano escluse nella selezione delle regioni contigue
# TODO perche non raggiungono l'area minima
for i in range(0, numLabels):
    # if this is the first component then we examine the
    # *background* (typically we would just ignore this
    # component in our loop)
    if i == 0:
        text = "examining component {}/{} (background)".format(i + 1, numLabels)
        # otherwise, we are examining an actual connected component
    else:

        area = stats[i, cv.CC_STAT_AREA]

        if area > 350:
            text = "examining component {}/{}".format(i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))

            print("Area: {}".format(area))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]

            # TODO è giusto?
            if (x - erosionIt) >= 0:
                x = x - erosionIt
            if (y - erosionIt) >= 0:
                y = y - erosionIt
            if (x + w + erosionIt * 2) <= c:
                w = w + erosionIt * 2
            if (y + h + erosionIt * 2) <= r:
                h = h + erosionIt * 2

            (cX, cY) = centroids[i]




            # mask = np.zeros((h,w), dtype="uint8")
            componentMaskBool = (labels[y:y + h, x:x + w] == i).astype("uint8")

            cv.imshow('test', componentMaskBool)
            # TODO metterla fuori dal ciclo
            componentMaskBool = open_close(componentMaskBool, 'open', 3, er_it=0, dil_it=erosionIt, shape=kShape)
            componentMaskBool = open_close(componentMaskBool, 'close', 3, er_it=erosionIt, dil_it=erosionIt,
                                           shape=kShape)

            componentMask = componentMaskBool * 255

            masked = cv.bitwise_and(image[y:y + h, x:x + w], image[y:y + h, x:x + w], mask=componentMask)

            gray = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)

            #mult = area / 40000

            mult = 0.2 # TODO tune

            r = math.sqrt(area / math.pi) * mult
            print("Radius: {}".format(r))

            lbpDescriptor = LocalBinaryPatterns(18, r , method = "uniform")

            lbp = lbpDescriptor.describe(componentMask, componentMaskBool.astype("bool"))
            # x.compute_lbp(componentMask, componentMaskBool.astype("bool"))

            #Eseguo la predizione
            prediction = m.predict_with_best_n([lbp], 5)
            display(prediction)
            category = prediction['Prediction'].iat[0]


            #Disegno la bounding box e il centroide
            cv.rectangle(rectImage, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.circle(rectImage, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            cv.putText(rectImage, category, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)



            cv.imshow('componentMask %d' % i, componentMask)
            cv.imshow('masked %d' % i, masked)

cv.imshow('bounding box', rectImage)
cv.waitKey(0)

exit(0)

# DONE trova altro modo per normalizzare luce(con gamma)
# DONE slider con precisione floating point
# TODO rendi tutto una funzione, in particolare ogni layer della pipeline

# TODO implementa la possibbilità di aggiungere più immagini di bg
# TODO? implementa l'informazione sugli oggetti
# TODO rumore gaussiano
# TODO sperimentare con altri spazi colore


# TODO Funzione che divide immagine in tasselli
# TODO Altri descrittori

# TODO LBP e corner detection con istogrammi
# TODO Segmentazione
# TODO Guarda scala

# TODO shape detection con humoments
# TODO skeleton
# TODO sottrai
# TODO regioni convesse e simmetriche

# TODO Classificazione

# questo ci può servire per fare la sottrazione di due immagini più avanti nel progetto
# alpha = val / alpha_slider_max
# beta = ( 1.0 - alpha )
# dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
