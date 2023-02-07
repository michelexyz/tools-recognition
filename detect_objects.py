from matplotlib import pyplot as plt

from descriptors import color_descriptors as dsc

from extract_objects import extract_with_trackbar
from segmentation import segment_and_detect
from descriptors.shape_descriptors import hu_fun

from gamma import normalize_with_trackbar, optimal_gamma_on_intensity

from parameters import *







def detect_objects(image, bg, descriptor, obj_classifier=None, bgClassifier=None ):


    # Mostra immagine con oggetti multipli
    cv.imshow("original image", image)

    cv.waitKey(0)

    # Mostra immagine di background
    cv.imshow("original bg", bg)

    #Applica una gamma selezionata dall'utente all'immagine
    #E memorizza l'intensità media risultante in mean_perc
    norm_im, mean_perc = normalize_with_trackbar(image)

    #analogamente applica una gamma all'immagine di sfondo per raggiungere la stessa intensità media
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

    #Binarizza l'immagine dato il colore medio del background e la sua deviazione standard
    obj_thresh = extract_with_trackbar(norm_im, bgRgbMean, bgRgbStd)


    #salva l'immagine con le bounding box della predizione in rectImage
    #salva l'immagine segmentata in labels
    rectImage,(_,labels,_,_,_) = segment_and_detect(image, obj_thresh, obj_classifier,descriptor)

    plt.imshow(labels)
    plt.colorbar()
    plt.show()


    cv.imshow('bounding box', rectImage)
    cv.waitKey(0)

    return rectImage




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
