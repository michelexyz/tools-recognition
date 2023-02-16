import sys
from pathlib import Path
#from rembg import remove, new_session
import os
import numpy as np

from descriptors import DescriptorInterface
from descriptors.shape_descriptors import CallableHu
from descriptors.texture_descriptors import CallableLbp
from tr_utils.useful import resize_to_fixed_d, remove_imperfections_adv
from tr_utils.parameters import *

from files.files_handler import get_file_abs_path


def describe_data(descriptor: DescriptorInterface, dataset_path="/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/processed", output_file='data.npy', draw= True):
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
    dim, dims = descriptor.get_dim()#TODO

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

                print(f'Analisi immagine: {file.stem}')

                print('Original Dimensions : ', input.shape)


                # width = 500
                # height = int(width * (input.shape[0]/input.shape[1]))
                # #TODO scala a larghezza fissa
                # scale_percent = 50  # percent of original size
                # #width = int(input.shape[1] * scale_percent / 100)
                # #height = int(input.shape[0] * scale_percent / 100)
                # dim = (width, height)
                #
                # # resize image
                # input = cv.resize(input, dim, interpolation=cv.INTER_AREA)


                # input_min = resize_to_fixed_d(input, 64)
                # _, _, _, Amin = cv.split(input_min)



                B, G, R, A = cv.split(input)
                #cv.imshow('RED {}, {}'.format(i, category), R)


                #Immagine binarizzata
                binarized_bool = (A > 0).astype("uint8")

                is_searched_file = file.stem == 'IMG_6821.out'
                #Rimuovi le imperfezioni (open e close) e estrai solo la regione connessa più grande


                if is_searched_file:
                    cv.imshow(f'original mask {file.stem}, {category}', binarized_bool.astype("uint8") * 255)

                # remove small objects per escludere gi eventuali oggetti separati
                # e prima della resize per pulire l'immagine con più precisione

                # binarized_bool = remove_small_objects(binarized_bool)
                #
                # if is_searched_file:
                #     cv.imshow(f'mask without small obj {file.stem}, {category}', binarized_bool.astype("uint8") * 255)

                binarized_bool = remove_imperfections_adv(binarized_bool)
                if is_searched_file:
                    cv.imshow(f'mask without imperfections {file.stem}, {category}', binarized_bool.astype("uint8") * 255)




                binarized_bool = resize_to_fixed_d(binarized_bool, d)

                if is_searched_file:
                    cv.imshow(f'mask resized {file.stem}, {category}', binarized_bool.astype("uint8") * 255)


                print('Resized bool Dimensions : ', binarized_bool.shape)

                area = np.count_nonzero(binarized_bool)
                print("Area: {}".format(area))

                binarized = binarized_bool.astype("uint8") * 255
                binarized_bool = binarized_bool.astype("bool")


                # lbpDescriptor = parametric_lbp(P, method=method, area=area ,width = width)
                #
                #
                # desc = lbpDescriptor.describe(binarized, binarized_bool)
                # x.compute_lbp(componentMask, componentMaskBool.astype("bool"))

                desc = descriptor.describe(binarized, binarized_bool, area, file.stem, draw=draw)

                #print("ciao")

                X[element_index] = desc.reshape(-1)
                Y[element_index] = category_index
                names[element_index] = file.stem

                binarized_min = resize_to_fixed_d(binarized, 64)

                images[element_index] = binarized_min



                #cv.imshow('maschera {}, {}'.format(element_index, category), binarized)
                element_index   +=1
            category_index += 1
    data = np.empty(6, dtype=object)
    data[0] = X
    data[1] = Y
    data[2] = names
    data[3] = np.array([category_legend]) #TODO togli parentesi quadrate
    data[4] = images
    data[5] = np.array(dims)
    # save to  file
    np.save(output_file, data)



            # noBgImg = remove(input, session=session)
            #
            # # Contours,imgContours = cv.findContours(noBgImg,None , None)
            # [X, Y, W, H] = cv.boundingRect(cv.cvtColor(noBgImg, cv.COLOR_BGR2GRAY))
            # cropped_image = noBgImg[Y:Y + H, X:X + W]
            #
            # cv.imwrite(output_path, cropped_image)
            # print('generata ' + imgName)


if __name__ == '__main__':
    print("esecuzione di describe_data.py")

    hu = CallableHu()
    lbp = CallableLbp(P = P, method=method)
    output_file = get_file_abs_path('hu_data.npy')

    describe_data(hu, output_file=output_file)

#prepare_models()



    sys.exit()
#TODO numero iterazioni open e close proporzionale alla definizione dell'immagine
#TODO resize per normalizzare?
# In caso farlo nello script bg_remover