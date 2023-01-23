import math

import numpy as np
import cv2 as cv

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

# Trova la gamma data un'intesità media obbiettivo
def optimalGamma(goal_perc, mean):
    if goal_perc <= 0 or goal_perc >= 1:
        raise Exception("goal_mean fuori dal range (0,1) valore %d" % goal_perc)

    exp = math.log(goal_perc) / math.log(mean / 255)
    gamma = 1 / exp
    print(f"goal: {goal_perc}, mean: {mean}, gamma: {gamma}")
    return gamma

# Trova la gamma ottimale data un 'immagine a una media obbiettivo
# E la applica sul canale Intensità(V)
def optimal_gamma_on_intensity(im, goal_perc):
    # Estrai il valore di intesità medio dal bg
    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    _, _, V = cv.split(hsv)
    bgVMean = np.mean(V)

    # Normalizza l'mmagine di bg con la media obiettivo fornita dall'utente
    op = optimalGamma(goal_perc=goal_perc, mean=bgVMean)
    print("GAMMA OTTIMALE %f" % op)
    out = gammaIntensity(im, op)

    return out

def on_gamma(g, im):
    # cv.imshow("vediamo se on gamma to funziona ori", NormIm)
    # np.copyto(normIm, gammaIntensity(im, g / 10))
    out = gammaIntensity(im, g / 10)
    cv.imshow("gamma", out)


def on_mean(m_perc, mean):
    goal_m = m_perc / 100
    optimal_g = optimalGamma(goal_m, mean)
    cv.setTrackbarPos('gamma', 'gamma', int(optimal_g * 10))

# Data un'immagine
# Permette all'utente di selezionare la gamma desiderata con una trackbar
# il cui risultato è mostrato in tempo reale
# Ritorna l'immagine sulla quale è stata applicata l'operazione di gamme e
# L'intesità media della stessa
def normalize_with_trackbar(image):
    """

    :param image:
    :return:
    """
    # Estrai il valore di intesità medio dall' immagine da analizzare
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    _, _, V = cv.split(hsv)
    v_mean = np.mean(V)

    print("vmean: %d" % v_mean)

    # NormIm = normilezeIntesity(image, VMeanBg)
    # NormIm = equalizeIntensity(image)
    # Immagine normalizzata che sarà aggiornata dalle operazioni successive
    #NormIm = np.copy(image)

    # cv.imshow('NormIm', NormIm)

    # FINESTRA per l'alterazione della gamma
    cv.namedWindow("gamma")
    trackbar_gamma = 'gamma'
    trackbar_mean = 'mean'

    # imposta le trackbar per la gamma e l'intensità media obbiettivo
    cv.createTrackbar(trackbar_gamma, "gamma", 1, 100, lambda v: on_gamma(v, image))
    cv.createTrackbar(trackbar_mean, "gamma", 1, 100, lambda v: on_mean(v, v_mean))

    # Inizializza la l'intensità media obiettivo al 80%
    cv.setTrackbarPos(trackbar_mean, 'gamma', 80)

    # Attendi la selezione della gamma da parte dell'utente per passare alla fase successiva
    cv.waitKey(0)  # TODO

    # Estrai la gamma selezionata dalla posizione della trackbar e applicala all'imagine
    selected_gamma = cv.getTrackbarPos(trackbar_gamma, 'gamma') / 10


    NormIm = gammaIntensity(image, selected_gamma)

    # Memorizza l'intensità media obiettivo fornita dall'utente
    mean_perc = cv.getTrackbarPos(trackbar_mean, 'gamma') / 100

    # Ritorna l'immagine data in imput sulla quale è stata eseguita un operazione di gamma
    # e l'intensità media della stessa in percentuale
    return NormIm, mean_perc