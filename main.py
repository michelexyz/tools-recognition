import sys

import cv2 as cv

from descripted_data_analysis.category_description_analysis import category_dsc_analysis
from descripted_data_analysis.compute_mean import load_mean_and_std
from descripted_data_analysis.feature_extraction import analize_data
from descriptors import DescriptorCombiner
from descriptors.shape_descriptors import CallableHu
from descriptors.texture_descriptors import CallableLbp
from detection.detect_objects import detect_objects

import numpy as np

from prepare_data import describe_data

from descripted_data_analysis.scale_data import scale_data
from tr_utils import parameters
from classification.prepare_models import prepare_models

from files.files_handler import get_file_abs_path, get_processed_path, get_data_folder



# TODO gestione eccezioni dei file
# TODO UI E analisi dei file specifici

print('yo')
# CREA I DESCRITTORI
hu = CallableHu()
lbp = CallableLbp(P=parameters.P, method=parameters.method)
print('yo')
# COMBINA I DESCRITTORI
descriptor = DescriptorCombiner([hu, lbp], [5, 1])

describe_str = 'describe'
category_an_str = 'category_an'
scale_str = 'scale'
extract_str = 'analyze'
train_str = 'train'
detect_str = 'detect'

default_pipe = [describe_str, category_an_str, train_str, detect_str]

# pipe_line = [describe_str]

pipe_line = default_pipe

pipe_line = [category_an_str, scale_str, train_str, detect_str]

# pipe_line = [scale_str, train_str, detect_str]

# pipe_line = [train_str, detect_str]

# pipe_line = [detect_str]

# feature_extracted = False

f_extraction_data = None
scaling_data = None
with_extracted_data = False
with_scaled_data = False

if extract_str in pipe_line:
    with_extracted_data = True

if scale_str in pipe_line:
    with_scaled_data = True

im_name = 'tools.jpg'
#im_name = 'IMG_7158.jpeg'
#im_name = 'IMG_7084.jpeg'

dataset_path = str(get_processed_path())

bg_path = str(get_data_folder('backgrounds').joinpath('green', 'green.jpg').resolve())

#bg_path = str()

im_path = str(get_data_folder('multiple_objects').joinpath(im_name).resolve())

#im_path = str(get_data_folder('non_omogeneus_backgrounds').joinpath('red_square','IMG20230120123359.jpg' ).resolve())

op_num = 1

draw_dataset = False
draw_test = False

described_path = str(get_file_abs_path('data.npy'))
scaled_path = str(get_file_abs_path('data_scaled.npy'))
extracted_path = str(get_file_abs_path('data_extracted.npy'))

cl_path = str(get_file_abs_path('cl1.npy'))
cl_scaled_path = str(get_file_abs_path('cl1_scaled.npy'))
cl_extracted_path = str(get_file_abs_path('cl1_extracted.npy'))

extraction_path = str(get_file_abs_path('extraction_data.npy'))
scale_path = str(get_file_abs_path('scale_data.npy'))

if describe_str in pipe_line:
    print(f"Operazione {op_num}: DESCRIZIONE DEI DATI")
    op_num += 1

    # DESCRIVE LE IMMAGINI E SALVA I DATI SU FILE
    describe_data(descriptor, dataset_path_str=dataset_path, output_file=described_path, draw=draw_dataset)

if category_an_str in pipe_line:
    print(f"Operazione {op_num}: ANALISI PER CATEGORIA")
    op_num += 1
    load_mean_and_std(described_path, descriptor)
    category_dsc_analysis(described_path, descriptor)

if scale_str in pipe_line:
    print(f"Operazione {op_num}: SCALING")
    op_num += 1
    scale_data(described_path, scaled_path, scale_path, weights=descriptor.weights)

if extract_str in pipe_line:
    print(f"Operazione {op_num}: ANALISI DEI DATI E FEATURE EXTRACION")
    op_num += 1

    _, _ = analize_data(scaled_path, extracted_path, extraction_path, draw=True)

if train_str in pipe_line:
    print(f"Operazione {op_num}: TRAINING DEL MODELLO")
    op_num += 1

    # ALLENA I MODELLI E LI SALVA SU FILE

    if with_extracted_data:
        prepare_models(extracted_path, cl_extracted_path)
    elif with_scaled_data:
        print("Alleno il modello con i dati scalati")
        prepare_models(scaled_path, cl_scaled_path)
    else:
        prepare_models(described_path, cl_path)

if detect_str in pipe_line:
    print(f"Operazione {op_num}: OBJECT DETECTION")
    op_num += 1
    # Leggi immagine con oggetti multipli
    image = cv.imread(im_path)

    # Leggi immagine del relativo background
    bg = cv.imread(bg_path)

    # CARICA IL MODELLO
    m = None
    if with_extracted_data:
        cl_data = np.load(cl_extracted_path, allow_pickle=True)

        m = cl_data[0]

        # CARICA I DATI DI FEATURE EXTRACTION
        data = np.load(extraction_path, allow_pickle=True)
        f_extraction_data = data[0]
        # CARICA I DATI DI SCALING
        scale_data_p = np.load(scale_path, allow_pickle=True)
        scaling_data = scale_data_p[0]


    elif with_scaled_data:
        print("Carico il modello allenato con i dati scalati")
        cl_data = np.load(cl_scaled_path, allow_pickle=True)

        m = cl_data[0]
        # CARICA I DATI DI SCALING
        scale_data_p = np.load(scale_path, allow_pickle=True)
        scaling_data = scale_data_p[0]

    else:
        cl_data = np.load(cl_path, allow_pickle=True)
        m = cl_data[0]

    detect_objects(image, bg, obj_classifier=m, descriptor=descriptor,
                   transformation_data=(scaling_data, f_extraction_data),
                   draw=draw_test)


sys.exit(0)
