from matplotlib import pyplot as plt

from binarization.binarize_with_bg import binarize_with_bg
from detection.segment_and_detect import segment_and_detect

from tr_utils.parameters import *






def detect_objects(image, bg, descriptor, obj_classifier=None, bgClassifier=None , transformation_data= None, draw = True):



    obj_thresh = binarize_with_bg(image, bg)

    #salva l'immagine con le bounding box della predizione in rectImage
    #salva l'immagine segmentata in labels
    rectImage,(_,labels,_,_,_) = segment_and_detect(image, obj_thresh, obj_classifier,descriptor, transformation_data, draw=draw)

    plt.imshow(labels)
    plt.colorbar()
    plt.show()


    cv.imshow('bounding box', rectImage)
    cv.waitKey(0)

    return rectImage




# DONE trova altro modo per normalizzare luce(con gamma)
# DONE slider con precisione floating point
# DONE 0 rendi tutto una funzione, in particolare ogni layer della pipeline
# TODO 0 funzione per capire quile file sono previsti in modo sbagliato nel test

# TODO implementa la possibbilità di aggiungere più immagini di bg
# DONE implementa l'informazione sugli oggetti
# TODO rumore gaussiano
# TODO sperimentare con altri spazi colore


# TODO Funzione che divide immagine in tasselli
# TODO Altri descrittori

# DONE LBP e corner detection con istogrammi
# DONE Segmentazione
# DONE Guarda scala

# DONE shape detection con humoments
# TODO skeleton
# TODO sottrai
# TODO regioni convesse e simmetriche

# DONE Classificazione

# DONE pulisci dati
# DONE prendi 5 migliori previsioni con pesi
# TODO min rect e Moemnti
# TODO vedi la dimensione giusta dell'lbp ror
# DONE approsimazione int open-close




# questo ci può servire per fare la sottrazione di due immagini più avanti nel progetto
# alpha = val / alpha_slider_max
# beta = ( 1.0 - alpha )
# dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
