import cv2 as cv
import numpy as np


# Binarizza l'immagine data la media dei valori e la deviazione std di un'immagine di bg
def extract_objects(im, k, channel_mean: np.ndarray, std: np.ndarray):
    if channel_mean.shape != std.shape:
        raise Exception("dimensione dell'array della media di canali diversa da quello della std")
    lower_bound = channel_mean - (std * k)
    upper_bound = channel_mean + (std * k)
    bg_threshold = cv.inRange(im, lower_bound, upper_bound)
    obj_threshold = cv.bitwise_not(bg_threshold)
    return obj_threshold

def extract_with_trackbar(im, channel_mean, std):
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
        obj_thresh = extract_objects(im, val, channel_mean, std)
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
    cv.waitKey(0) #TODO
    k = cv.getTrackbarPos(trackbar_k, bin_window)

    obj_thresh = extract_objects(im, k, channel_mean, std)
    cv.imwrite('no_bg_gamma.jpg', obj_thresh)
    return obj_thresh