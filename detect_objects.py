import math

import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

import descriptors as dsc

import texture_descriptors as tx
from extract_objects import extract_with_trackbar
from segmentation import segment_and_detect
from texture_descriptors import parametric_lbp
from texture_descriptors import parametric_lbp
from texture_descriptors import find_r_mode

from texture_descriptors import LocalBinaryPatterns

from gamma import normalize_with_trackbar, optimal_gamma_on_intensity
from morph_fun import open_close

from models import Models

import pandas as pd
from IPython.display import display

from useful import remove_imperfections

from parameters import *



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



def detect_objects(image, bg, bgClassifier=None, objClassifier=None):


    # Mostra immagine con oggetti multipli
    cv.imshow("original image", image)

    cv.waitKey(0)

    # Mostra immagine di background
    cv.imshow("original bg", bg)

    #Applica una gamma selezionata dall'utente all'immagine
    #E memorizza l'intensità media risultante in mean_perc
    norm_im, mean_perc = normalize_with_trackbar(image)

    #
    NormBg = optimal_gamma_on_intensity(bg, goal_perc=mean_perc)
    cv.imshow("normilized bg", NormBg)

    # Mostra anche l'immagine da analizzare normalizzata
    cv.imshow("normilized image", norm_im)

    # calcola i valori BGR medi dell'immagine di bg
    bgRgbMean = dsc.computeColor(NormBg)[0]
    print(bgRgbMean)

    # calcola la dev standard dell'immagine di bg
    bgRgbStd = dsc.computeStd(NormBg)[0]
    std = dsc.computeStd(image)[0]

    print(std)
    print(bgRgbStd)

    obj_thresh = extract_with_trackbar(norm_im, bgRgbMean, bgRgbStd)

    # CARICA IL MODELLO
    data = np.load('cl1.npy', allow_pickle=True)

    m = data[0]

    def lbp_fun(componentMask, componentMaskBool, area):
        lbpDescriptor = parametric_lbp(P, method=method, area=area)
        lbp = lbpDescriptor.describe(componentMask, componentMaskBool)
        return lbp


    rectImage,(_,labels,_,_,_) = segment_and_detect(image, obj_thresh,m,lbp_fun)

    plt.imshow(labels)
    plt.colorbar()
    plt.show()


    cv.imshow('bounding box', rectImage)
    cv.waitKey(0)

    return rectImage


# Leggi immagine con oggetti multipli
image = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/tools.jpg"))

# Leggi immagine del relativo background
bg = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/green.jpg"))

detect_objects(image,bg)

# DONE trova altro modo per normalizzare luce(con gamma)
# DONE slider con precisione floating point
# TODO 0 rendi tutto una funzione, in particolare ogni layer della pipeline
# TODO 0 funzione per capire quile file sono previsti in modo sbagliato nel test

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

# TODO pulisci dati
# DONE prendi 5 migliori previsioni con pesi
# TODO min rect e Moemnti
# TODO vedi la dimensione giusta dell'lbp ror
# TODO approsimazione int open-close




# questo ci può servire per fare la sottrazione di due immagini più avanti nel progetto
# alpha = val / alpha_slider_max
# beta = ( 1.0 - alpha )
# dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
