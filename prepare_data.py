import sys
from pathlib import Path
#from rembg import remove, new_session
import os
import numpy as np

from descripted_data_analysis.compute_mean import compute_mean_and_std
from descriptors import DescriptorInterface
from descriptors.shape_descriptors import CallableHu
from descriptors.texture_descriptors import CallableLbp
from tr_utils.useful import resize_to_fixed_d, remove_imperfections_adv
from tr_utils.parameters import *

from files.files_handler import get_file_abs_path


def describe_data(descriptor: DescriptorInterface, dataset_path_str, output_file='data.npy', draw= True):


    dataset_path = Path(dataset_path_str)
    # Finds all the files in the dataset folder
    search_path = Path('**')/'*.png'
    all = dataset_path.glob(str(search_path))
    # Counts them
    n_samples = len(list(all))
    # dim = LocalBinaryPatterns().compute_dim(P, method)
    dim, dims = descriptor.get_dim()#TODO

    # Allocate memory for data to be saved
    data = np.empty(8, dtype=object)
    X = np.empty((n_samples, dim))
    Y = np.empty(n_samples)
    file_names = np.empty(n_samples, dtype=object) # Names of sample files
    category_legend = []
    images = np.empty(n_samples, dtype=object)# Image previews of samples
    sample_per_category = []

    category_index = 0
    element_index = 0

    # Iterates on every category folder
    for categoryPath in dataset_path.iterdir():
        if categoryPath.is_dir():
            category = categoryPath.stem

            if category != '.DS_Store':
                category_legend.append(category)

                # Variable to pick only one sample per-category
                sample_picked = False
                for file in categoryPath.glob('*.png'):
                    print(f'Analisi immagine: {file.stem}')

                    # Specifies a file for which to draw descriptors data
                    is_searched_file = file.stem == 'IMG_6821.out'

                    #Extract binarized and rgb masks
                    binarized, binarized_bool, masked = extract_sample(str(file),file.stem, category,is_searched_file)

                    desc = descriptor.describe(binarized, binarized_bool,name=file.stem, draw=draw)

                    X[element_index] = desc.reshape(-1)
                    Y[element_index] = category_index
                    file_names[element_index] = file.stem

                    binarized_min = resize_to_fixed_d(binarized, 64)

                    images[element_index] = binarized_min
                    if not sample_picked:
                        sample_per_category.append((binarized,binarized_bool, masked, category))
                        sample_picked = True



                    #cv.imshow('maschera {}, {}'.format(element_index, category), binarized)
                    element_index   +=1
                category_index += 1

    data[0] = X
    data[1] = Y
    data[2] = file_names
    data[3] = np.array([category_legend]) #TODO togli parentesi quadrate
    data[4] = images
    data[5] = np.array(dims)
    data[6] = np.array(sample_per_category)

    #means, stds = compute_mean_and_std(X, Y, descriptor)
    # Empty for means and stds
    data[7] = None

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



def extract_sample(input_path, name, category_name, draw = False):


    # imgName = (file.stem + ".out.png")
    # output_path = str(processedFolderStr + '/' + category + '/' + imgName)

    # Carica l'immagine nei 4 canali
    input = cv.imread(input_path, cv.IMREAD_UNCHANGED)
    rgb_image = cv.imread(input_path)

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
    # cv.imshow('RED {}, {}'.format(i, category), R)

    # BINARIZE IMAGE
    binarized_bool = (A > 0).astype("uint8")
    # Rimuovi le imperfezioni (open e close) e estrai solo la regione connessa pi?? grande

    if draw:
        cv.imshow(f'original mask {name}, {category_name}', binarized_bool.astype("uint8") * 255)

    # remove small objects per escludere gi eventuali oggetti separati
    # e prima della resize per pulire l'immagine con pi?? precisione

    # binarized_bool = remove_small_objects(binarized_bool)
    #
    # if is_searched_file:
    #     cv.imshow(f'mask without small obj {file.stem}, {category}', binarized_bool.astype("uint8") * 255)

    # TRANSFORM OPERATIONS
    binarized_bool, image = remove_imperfections_adv(binarized_bool, rgb_image)
    if draw:
        cv.imshow(f'mask without imperfections {name}, {category_name}', binarized_bool.astype("uint8") * 255)

    binarized_bool = resize_to_fixed_d(binarized_bool, d)
    image = resize_to_fixed_d(image, d)

    if draw:
        cv.imshow(f'mask resized {name}, {category_name}', binarized_bool.astype("uint8") * 255)

    print('Resized bool Dimensions : ', binarized_bool.shape)

    # BINARIZATION TRANSFORMED

    binarized = binarized_bool.astype("uint8") * 255
    binarized_bool = binarized_bool.astype("bool")

    masked = cv.bitwise_and(image, image, mask=binarized)
    return binarized, binarized_bool, masked

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