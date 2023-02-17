

import cv2 as cv

from descriptors.color_descriptors import computeColor, computeStd
from binarization.extract_objects import extract_with_trackbar
from tr_utils.gamma import normalize_with_trackbar, optimal_gamma_on_intensity


#BINARIZZA L'immagine attraverso un immagine campione dello sfondo
def binarize_with_bg(image, bg):
    # Mostra immagine con oggetti multipli
    cv.imshow("original image", image)

    cv.waitKey(0)

    # Mostra immagine di background
    cv.imshow("original bg", bg)

    # Applica una gamma selezionata dall'utente all'immagine
    # E memorizza l'intensità media risultante in mean_perc
    norm_im, mean_perc = normalize_with_trackbar(image)

    # analogamente applica una gamma all'immagine di sfondo per raggiungere la stessa intensità media
    NormBg = optimal_gamma_on_intensity(bg, goal_perc=mean_perc)
    cv.imshow("normilized bg", NormBg)

    # Mostra anche l'immagine da analizzare normalizzata
    cv.imshow("normilized image", norm_im)

    # calcola i valori BGR medi dell'immagine di bg
    bgRgbMean = computeColor(NormBg)[0]
    print(bgRgbMean)

    # calcola la dev standard dell'immagine di bg
    bgRgbStd = computeStd(NormBg)[0]
    std = computeStd(image)[0]

    print(std)
    print(bgRgbStd)

    # Binarizza l'immagine dato il colore medio del background e la sua deviazione standard
    obj_thresh = extract_with_trackbar(norm_im, bgRgbMean, bgRgbStd)
    return obj_thresh