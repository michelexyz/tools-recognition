import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

import descriptors as dsc


#NormilizeIntensity imposta ad ogni pixlel un Immagine BGR "Im"
#lo stesso valore di intensità "i_value"
#potrebbe essere anche direttamente creata una funzione "normalizeChannel"
#TODO METTI in file UTIL

def normilezeIntesity(im, i_value):
    hsvImage = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    HIm, SIm, VIm = cv.split(hsvImage)
    c, r, ch = im.shape


    VNormalized = np.full((c, r), i_value, np.uint8)
    hsvImage = cv.merge([HIm, SIm, VNormalized])
    NormImg = cv.cvtColor(hsvImage, cv.COLOR_HSV2BGR)
    return NormImg


#Leggi immagine con oggetti multipli
image = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/tools.jpg"))

#Leggi immagine del relativo background
bg = cv.imread(os.path.expanduser("~/PycharmProjects/ToolsRecognition/data/green.jpg"))

#Estrai valore di intensità medio dalla stessa
hsvBg = cv.cvtColor(bg, cv.COLOR_BGR2HSV)
_, _, VMeanBg = dsc.avarageColor(hsvBg)


NormIm = normilezeIntesity(image, VMeanBg)
cv.imshow("yooo",NormIm)

NormBg = normilezeIntesity(bg, VMeanBg)
cv.imshow("yooo",NormBg)

k = 1
#calcola i valori BGR medi dell'immagine di bg
bgRgbMean = dsc.computeColor(NormBg)[0]
BMean = bgRgbMean[0]
GMean = bgRgbMean[1]
RMean = bgRgbMean[2]
print(bgRgbMean)

#calcola la dev standard dell'immagine di bg
bgRgbStd = dsc.computeStd(NormBg)[0]
BStd = bgRgbStd[0]
GStd = bgRgbStd[1]
RStd = bgRgbStd[2]
print(bgRgbStd)

alpha_slider_max = 10
title_window = 'Immagine con slider'

def on_trackbar(val):
    k = val
    #TODO trasforma in una funzione il codice sotto
    lowBg = (BMean - BStd*k, GMean -  GStd*k, RMean - RStd*k)
    highBg = (BMean + BStd*k, GMean +  GStd*k, RMean + RStd*k)
    bg_threshold = cv.inRange(NormIm, lowBg, highBg)
    obj_threshold = cv.bitwise_not(bg_threshold)
    cv.imshow(title_window, obj_threshold)
    cv.imwrite("no_bg.jpg", obj_threshold)


cv.namedWindow(title_window)
trackbar_name = 'K slider %d' % alpha_slider_max
cv.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)
# Show some stuff
on_trackbar(1)


cv.waitKey(0)

exit(0)


#TODO trova altro modo per normalizzare luce(con gamma)
#TODO slider con precisione floating point
#TODO implementa la possibbilità di aggiungere più immagini di bg
#TODO? implementa l'informazione sugli oggetti

#TODO Funzione che divide immagine in tasselli

#questo ci può servire per fare la sottrazione di due immagini più avanti nel progetto
# alpha = val / alpha_slider_max
# beta = ( 1.0 - alpha )
# dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)