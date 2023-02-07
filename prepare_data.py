import sys
from pathlib import Path
#from rembg import remove, new_session
import os
import numpy as np

from descriptors import DescriptorInterface
from descriptors.shape_descriptors import hu_fun, CallableHu
from descriptors.texture_descriptors import CallableLbp
from train import prepare_models
from useful import remove_imperfections, remove_small_objects, resize_to_fixed_d
from parameters import *


def prepare_data(descriptor: DescriptorInterface, dataset_path="/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/processed"):
    #session = new_session()
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


    all = Path(dataset_path).glob('**/*.png')
    l = len(list(all))
    #dim = LocalBinaryPatterns().compute_dim(P, method)
    dim = descriptor.get_dim()#TODO

    X = np.empty((l, dim))
    Y = np.empty(l)
    names = np.empty(l, dtype=object)
    category_legend = []
    images = np.empty(l, dtype=object)

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
            category_legend.append(category)
            for file in categoryPath.glob('*.png'):
                input_path = str(file)

                #imgName = (file.stem + ".out.png")
                #output_path = str(processedFolderStr + '/' + category + '/' + imgName)

                #Carica l'immagine nei 4 canali
                input = cv.imread(input_path, cv.IMREAD_UNCHANGED)

                print('Original Dimensions : ', input.shape)


                width = 500
                height = int(width * (input.shape[0]/input.shape[1]))
                #TODO scala a larghezza fissa
                scale_percent = 50  # percent of original size
                #width = int(input.shape[1] * scale_percent / 100)
                #height = int(input.shape[0] * scale_percent / 100)
                dim = (width, height)

                # resize image
                input = cv.resize(input, dim, interpolation=cv.INTER_AREA)

                # input_min = resize_to_fixed_d(input, 64)
                # _, _, _, Amin = cv.split(input_min)

                print('Resized Dimensions : ', input.shape)

                B, G, R, A = cv.split(input)
                #cv.imshow('RED {}, {}'.format(i, category), R)


                #Immagine binarizzata
                binarized_bool = (A > 0).astype("uint8")

                #Rimuovi le imperfezioni (open e close) e estrai solo la regione connessa più grande
                binarized_bool = remove_imperfections(binarized_bool)
                binarized_bool = remove_small_objects(binarized_bool)

                area = np.count_nonzero(binarized_bool)
                print("Area: {}".format(area))

                binarized = binarized_bool.astype("uint8") * 255
                binarized_bool = binarized_bool.astype("bool")


                # lbpDescriptor = parametric_lbp(P, method=method, area=area ,width = width)
                #
                #
                # desc = lbpDescriptor.describe(binarized, binarized_bool)
                # x.compute_lbp(componentMask, componentMaskBool.astype("bool"))

                desc = descriptor.describe(binarized, binarized_bool, area)

                #print("ciao")

                X[element_index] = desc.reshape(-1)
                Y[element_index] = category_index
                names[element_index] = file.stem

                binarized_min = resize_to_fixed_d(binarized, 64)

                images[element_index] = binarized_min



                cv.imshow('maschera {}, {}'.format(element_index, category), binarized)
                element_index+=1
            category_index += 1
    data = np.empty(5, dtype=object)
    data[0] = X
    data[1] = Y
    data[2] = names
    data[3] = np.array([category_legend])
    data[4] = images
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


hu = CallableHu(draw=True)
lbp = CallableLbp(P = P, method=method)
prepare_data(hu)

prepare_models()

cv.waitKey(0)

sys.exit()
#TODO numero iterazioni open e close proporzionale alla definizione dell'immagine
#TODO resize per normalizzare?
# In caso farlo nello script bg_remover