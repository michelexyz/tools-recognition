import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display_functions import display

from tr_utils.morph_fun import open_close
from tr_utils.useful import remove_imperfections, resize_to_fixed_d

from tr_utils.parameters import *


def segment_and_detect(image, obj_thresh, m, descriptor, transformation_data = None, apply_on_mask = True, draw = True,show_objects= False, ):

    global kShape  # TODO add to parameters.py
    erosionIt = 2  # TODO tune
    # La dimensione del kernel deve essere = 3 per il funzionamento del programma(vedi dentro if area..)

    #obj_thresh = open_close(obj_thresh, 'close', 3, er_it=3, dil_it=3, shape=kShape)  # todo

    obj_thresh = open_close(obj_thresh, 'open', 3, er_it=erosionIt, dil_it=0, shape=kShape)

    # Segmentazione per componenti connesse
    global connectivity
    output = cv.connectedComponentsWithStats(obj_thresh, connectivity, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output

    new_num_labels = 0
    new_labels = np.zeros_like(labels)
    new_stats = []
    new_centroids = []
    predictions= []

    # Mostra l'immagine segmaentata
    plt.imshow(labels)
    plt.show

    # clone our original image (so we can draw on it) and then draw
    # a bounding box surrounding the connected component along with
    # a circle corresponding to the centroid
    rectImage = image.copy()
    r, c, _ = image.shape



    # loop over the number of unique connected component labels
    # DONE erode prima di estrazione dei singoli componenti per separare le "macche"
    # DONE E fare in modo che vengano escluse nella selezione delle regioni contigue
    # DONE perche non raggiungono l'area minima

    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(i + 1, numLabels)
            # Area will be inexact
            new_stats.append(stats[i])
            new_centroids.append(centroids[i])
            predictions.append("sfondo")
            # otherwise, we are examining an actual connected component

        else:

            area = stats[i, cv.CC_STAT_AREA]

            if area > 350:
                text = "examining component {}/{}".format(i + 1, numLabels)
                # print a status message update for the current connected
                # component
                print("[INFO] {}".format(text))

                # extract the connected component statistics and centroid for
                # the current label
                x = stats[i, cv.CC_STAT_LEFT]
                y = stats[i, cv.CC_STAT_TOP]
                w = stats[i, cv.CC_STAT_WIDTH]
                h = stats[i, cv.CC_STAT_HEIGHT]

                # DONE è giusto?
                # Modifica le dimensioni della bounding box in previsione di quanto l'oggetto sarò dilatao
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


                #Dilates to compensate effect of erosiom before segmentation
                componentMaskBool = open_close(componentMaskBool, 'open', 3, er_it=0, dil_it=erosionIt, shape=kShape)

                segmentedMaskBool = remove_imperfections(componentMaskBool)

                # componentMaskBool = open_close(componentMaskBool, 'close', 3, er_it=erosionIt, dil_it=erosionIt,
                #                                shape=kShape)
                area = np.count_nonzero(segmentedMaskBool)
                # Aggiungi dati per la nuova segmentazione
                new_num_labels += 1
                #new_labels[y:y + h, x:x + w] = componentMaskBool.astype("int32") * new_num_labels
                np.putmask(new_labels[y:y + h, x:x + w], segmentedMaskBool.astype("bool"), segmentedMaskBool.astype("int32") * new_num_labels)
                new_stats.append(stats[i])
                new_stats[new_num_labels][cv.CC_STAT_AREA] = area
                new_centroids.append(centroids[i])


                #Dimensione dell'immagine alla quale effettuare la resize
                d = 500



                #Prima della resize per pulire l'oggetto con più precisione
                componentMaskBool = remove_imperfections(componentMaskBool)

                componentMaskBool = resize_to_fixed_d(componentMaskBool, d)
                resizedImage = resize_to_fixed_d(image[y:y + h, x:x + w], d)

                # Calcola l'area effettiva della componente connessa in seguito all'operazione di resize
                area = np.count_nonzero(componentMaskBool)

                print("Area: {}".format(area))

                # Calcolo la maschera con i valori da 0 a 255
                componentMask = componentMaskBool * 255

                # Applico la maschera all'oggetto ritagliato dall'imagina originale
                masked = cv.bitwise_and(resizedImage, resizedImage, mask=componentMask)


                obj_name = f'obj {new_num_labels}'

                if apply_on_mask:
                    description = descriptor.describe(componentMask, componentMaskBool.astype("bool"), name=obj_name,
                                                      draw=draw)
                else:
                    description = descriptor.describe(masked, componentMaskBool.astype("bool"), name=obj_name,
                                                      draw=draw)


                scale_data, f_extraction_data = transformation_data
                scaled = None
                if not(scale_data is None):

                    #reshape per scaler e pca
                    description = description.reshape((1,description.shape[0]))

                    dims, scaler, weights =scale_data
                    description = scaler.transform(description)
                    start = 0

                    for i, dim in enumerate(dims):
                        description[:, start: start + dim] = description[:,start: start + dim] * weights[i]

                        start += dim
                    scaled = description
                    description = scaled[0].reshape(-1)

                if not(f_extraction_data is None):
                    #Estraggo le feature data la PCA
                    pca, optimal_n = f_extraction_data
                    if scaled is None:
                        raise("cant' extract before scaling")
                    else:
                        extracted = pca.transform(scaled)

                        first_components = extracted[0][0:optimal_n]

                        description = first_components.reshape(-1)



                # Eseguo la predizione
                prediction = m.predict_with_best_n([description], len(m.classifiers))
                print("=" * 30)
                print(f" Predictions obj {new_num_labels}:")
                display(prediction)
                # category = prediction['Prediction'].iat[1]

                probabilistc_prediction = m.predict_with_multiple_proba([description], n=3)
                category = probabilistc_prediction[0]
                prob = probabilistc_prediction[1]

                #Aggiungi cateogoria all'array delle categorie predette
                predictions.append(category)

                # Disegno la bounding box e il centroide
                cv.rectangle(rectImage, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv.circle(rectImage, (int(cX), int(cY)), 4, (0, 0, 255), -1)
                cv.putText(rectImage, f'{obj_name}: {category}, prob: {prob}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                           (36, 255, 12), 2)

                if show_objects:

                    cv.imshow('componentMask %d' % new_labels, componentMask)
                    cv.imshow('masked %d' % new_num_labels, masked)


    segmentation = (new_num_labels, new_labels, np.array(new_stats), np.array(new_centroids), np.array(predictions))


    return rectImage, segmentation




# def segment(image, obj_thresh, show_objects= False): #TODO


