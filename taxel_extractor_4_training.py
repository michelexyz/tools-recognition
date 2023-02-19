# questo è per passare al trainer tasselli di sfondo o tasselli di oggetti
import os
from pathlib import Path

import glcm_descriptor
from tr_utils.useful import taxels_number
import numpy as np
from PIL import Image
import cv2 as cv
from tesselation import tassella_e_descrivi, tassella_e_descrivi_png


# prende in input path cartella dove ci sono cartelle di diversi bg bg,
# tassella e descrive tutte le immagini che ci sono,
# le salva in un array della forma [descrizioneTassello, label]
# dove in questo caso label è '0' (sfondo)
def bg_taxel_extractor(input_path, dim, step, descriptor):

    dataset_path = Path(input_path)
    print('path dataset letto')

    # how many backgrounds typologies there are
    bg_num = len([f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))])

    # inizializzo array dove ogni elemento è dimensione di dati di una tipologia di sfondo
    backgrounds_dim = np.empty(bg_num, dtype=int)
    # indice per accedere a "backgrounds"
    bg_type_index = 0

    # questo per beccare le dimensioni dei dati (quanti tasselli avrò in totale, per ogni tipo di sfondo)
    # itero sulle sottocartelle del dataset di sfondi
    for bg_type in dataset_path.iterdir():

        total_bg_taxels = 0
        # itero su ogni immagine di sfondo
        for bg in bg_type.glob('*.jpg'):

            # per ogni immagine, calcolo quanti tasselli escono.
            with Image.open(bg) as img:
                width, height = img.size

            # adding current taxels number, to total
            total_bg_taxels += taxels_number(width, height, dim, step)

        backgrounds_dim[bg_type_index] = total_bg_taxels
        print(f'bg type {bg_type_index} with size {total_bg_taxels}')
        bg_type_index += 1

    # ora posso inizializzare l' array dove effettivamente popolo con le descrizioni dei tasselli
    backgrounds_descriptions = np.empty(bg_num, dtype=object)

    # index for accessing background_description
    bg_type_index = 0
    # itero sulle sottocartelle del dataset di sfondi
    for bg_type in dataset_path.iterdir():

        # variable to manage
        first_iteration = True

        # itero su ogni immagine di sfondo
        for bg in bg_type.glob('*.jpg'):

            # leggo immagine in livelli di grigio
            img = cv.imread(str(bg), cv.IMREAD_GRAYSCALE)

            # get size
            height, width = img.shape
            features_number = 2

            # tassella e descrivi
            if first_iteration:
                description = tassella_e_descrivi(img, step, dim, features_number, descriptor)
            else:
                description = np.concatenate((description, tassella_e_descrivi(img, step, dim, features_number, descriptor)))

            first_iteration = False

        print(f'sfondo tassellato del tipo {bg_type_index} descritto')

        # array di label della dimensione dell' array di tasselli del bg_type corrente
        labels = np.zeros(description.shape[0], dtype=int)

        # concateno descrizione sfondo corrente e label '0' (sfondo)
        description_labelled = (description, labels)

        # salvo nell'indice giusto sta coppia che scoppia di (descrizione, label)
        backgrounds_descriptions[bg_type_index] = description_labelled

        # incrementing bg type... e daaaje cor prossimooooo
        bg_type_index += 1

    return backgrounds_descriptions


# come quella sopra, soltanto per gli attrezzi PNG
def fg_taxel_extractor(input_path, dim, step, descriptor):

    # input tools dir img path
    dataset_path = Path(input_path)

    search_path = Path('**') / '*.png'
    all = dataset_path.glob(str(search_path))
    # Counts them
    n_samples = len(list(all))
    # dim = LocalBinaryPatterns().compute_dim(P, method)
    features_number = 2

    category_index = 0
    element_index = 0

    # to account 'description' array concatenation
    first_iteration = True

    # iterate over different tools directories
    for tool_category in dataset_path.iterdir():

        if tool_category.is_dir():

            # getting pure file name (no path, no extension)
            category = tool_category.stem

            # to manage some macOS stuff...
            if category != '.DS_Store':

                print(f'Analisi dir {category}')

                # iterate on each tool image
                for file in tool_category.glob('*.png'):

                    print(f'Analisi immagine: {file.stem}')

                    # tassello e descrivo i tools.png
                    if first_iteration:
                        description = tassella_e_descrivi_png(str(file), step, dim, features_number, descriptor)
                    else:
                        description = np.concatenate(
                            (description, tassella_e_descrivi_png(str(file), step, dim, features_number, descriptor)))

                    first_iteration = False

    # put descripted image's taxels in X, and label it '1' (object) in Y

    # compute 'descriptions''s length to build Y
    x_length = description.shape[0]

    X = description
    Y = np.ones(x_length, dtype=int)

    # creating tuple of [TASSELLO_DESCRIPTION, LABEL] (label is 1)
    tools_descriptions = (X, Y)

    return tools_descriptions


# "MAIN"

patch_size = 30
step = 20

# BACKGROUND
# folder_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/data/only_background'
# bg_tasselli_descritti = bg_taxel_extractor(folder_path, patch_size, step, glcm_descriptor.glcm_descriptor)

# FOREGROUND
folder_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/data/processed'
tools_tasselli_descritti = fg_taxel_extractor(folder_path, patch_size, step, glcm_descriptor.glcm_descriptor)


save_bg_description = True

if save_bg_description:
    save_data_path = 'C:/Users/glauc/PycharmProjects/tools-recognition/files/tools_taxel_description/'
    output_file = f'tools_taxel_descripted_glcm_patch{patch_size}_step{step}.npy'
    np.save(save_data_path + output_file, tools_tasselli_descritti)
    print('descrizione salvata nel file')
