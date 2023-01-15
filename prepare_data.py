import math
import sys
from pathlib import Path
from rembg import remove, new_session
import os, shutil
import cv2 as cv
import numpy as np
import pandas as pd

from morph_fun import open_close
from texture_descriptors import LocalBinaryPatterns


def prepare_data(dataset_path="/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/processed"):
    session = new_session()
    #
    # rawFolderStr = '/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/raw'
    # rawFolder = Path(rawFolderStr)
    # #processedFolderStr = str(rawFolder.parent / 'processed')
    # processedFolderStr = '/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/processed'

    # for filename in os.listdir(processedFolderStr):
    #     file_path = os.path.join(processedFolderStr, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print('Failed to delete %s. Reason: %s' % (file_path, e))
    # print('file eliminati')
    print('ciao')
    print('1')
    print('2')
    print('3')
    method = "uniform"
    P = 18

    all = Path(dataset_path).glob('**/*.png')
    l = len(list(all))
    dim = LocalBinaryPatterns().compute_dim(P, method)

    X = np.empty((l, dim))
    Y = np.empty(l)
    names = np.empty(l, dtype=object)
    category_index = 0
    element_index = 0
    debug = os.listdir(dataset_path)

    for category in os.listdir(dataset_path):
        categoryPathStr = dataset_path + '/' + category
        categoryPath = Path(categoryPathStr)
        #processedCategoryPathStr = processedFolderStr + '/' + category
        #processedCategoryPath = Path(processedCategoryPathStr)
        #processedCategoryPath.mkdir(parents=True, exist_ok=True)
        if category != '.DS_Store':
            for file in categoryPath.glob('*.png'):
                input_path = str(file)

                #imgName = (file.stem + ".out.png")
                #output_path = str(processedFolderStr + '/' + category + '/' + imgName)

                #Carica l'immagine nei 4 canali
                input = cv.imread(input_path, cv.IMREAD_UNCHANGED)

                print('Original Dimensions : ', input.shape)

                #TODO scala a larghezza fissa
                scale_percent = 50  # percent of original size
                width = int(input.shape[1] * scale_percent / 100)
                height = int(input.shape[0] * scale_percent / 100)
                dim = (width, height)

                # resize image
                input = cv.resize(input, dim, interpolation=cv.INTER_AREA)

                print('Resized Dimensions : ', input.shape)

                B, G, R, A = cv.split(input)
                #cv.imshow('RED {}, {}'.format(i, category), R)


                binarized_bool = (A > 0).astype("uint8")
                binarized_bool = open_close(binarized_bool, 'close', 3, er_it=5, dil_it=5,
                                               shape=cv.MORPH_RECT)

                area = np.count_nonzero(binarized_bool)
                print("Area: {}".format(area))

                binarized = binarized_bool.astype("uint8") * 255
                binarized_bool = binarized_bool.astype("bool")

                mult = 0.2  # TODO tune

                r = math.sqrt(area / math.pi) * mult
                print("Radius: {}".format(r))

                lbpDescriptor = LocalBinaryPatterns(P, r, method=method)

                lbp = lbpDescriptor.describe(binarized, binarized_bool)
                # x.compute_lbp(componentMask, componentMaskBool.astype("bool"))

                #print("ciao")

                X[element_index] = lbp
                Y[element_index] = category_index
                names[element_index] = file.stem

                cv.imshow('maschera {}, {}'.format(element_index, category), binarized)
                element_index+=1
            category_index += 1
    data = np.empty(3, dtype=object)
    data[0] = X
    data[1] = Y
    data[2] = names
    # save to csv file
    np.save('data.npy', data)



            # noBgImg = remove(input, session=session)
            #
            # # Contours,imgContours = cv.findContours(noBgImg,None , None)
            # [X, Y, W, H] = cv.boundingRect(cv.cvtColor(noBgImg, cv.COLOR_BGR2GRAY))
            # cropped_image = noBgImg[Y:Y + H, X:X + W]
            #
            # cv.imwrite(output_path, cropped_image)
            # print('generata ' + imgName)


prepare_data()
cv.waitKey(0)

exit(0)
#TODO numero iterazioni open e close proporzionale alla definizione dell'immagine
#TODO resize per normalizzare?
# In caso farlo nello script bg_remover