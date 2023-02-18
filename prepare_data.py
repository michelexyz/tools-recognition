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


def describe_data(descriptor: DescriptorInterface, dataset_path_str="/Users/michelevannucci/PycharmProjects/ToolsRecognition/data/processed", output_file='data.npy', draw= True):
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

    dataset_path = Path(dataset_path_str)
    # Finds all the fles in the dataset folder
    search_path = Path('**')/'*.png'
    all = dataset_path.glob(str(search_path))
    # Counts them
    n_samples = len(list(all))
    #dim = LocalBinaryPatterns().compute_dim(P, method)
    dim, dims = descriptor.get_dim()#TODO

    data = np.empty(8, dtype=object)
    X = np.empty((n_samples, dim))
    Y = np.empty(n_samples)
    names = np.empty(n_samples, dtype=object)
    category_legend = []
    images = np.empty(n_samples, dtype=object)
    sample_per_category = []

    category_index = 0
    element_index = 0
    debug = os.listdir(dataset_path_str)

    for categoryPath in dataset_path.iterdir():
        # categoryPathStr = dataset_path_str + '/' + category
        # categoryPath = Path(categoryPathStr)

        #categoryPath = dataset_path / category
        if categoryPath.is_dir():
            category = categoryPath.stem
            #processedCategoryPathStr = processedFolderStr + '/' + category
            #processedCategoryPath = Path(processedCategoryPathStr)
            #processedCategoryPath.mkdir(parents=True, exist_ok=True)
            if category != '.DS_Store':
                category_legend.append(category)

                # Variable to pick only one sample per-category
                sample_picked = False
                for file in categoryPath.glob('*.png'):
                    print(f'Analisi immagine: {file.stem}')

                    is_searched_file = file.stem == 'IMG_6821.out'

                    binarized, binarized_bool, masked = extract_sample(str(file),file.stem, category,is_searched_file)



                    area = np.count_nonzero(binarized_bool)
                    print("Area: {}".format(area))


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
                    if not sample_picked:
                        sample_per_category.append((binarized,binarized_bool, masked, category))
                        sample_picked = True



                    #cv.imshow('maschera {}, {}'.format(element_index, category), binarized)
                    element_index   +=1
                category_index += 1

    data[0] = X
    data[1] = Y
    data[2] = names
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
    # Rimuovi le imperfezioni (open e close) e estrai solo la regione connessa più grande

    if draw:
        cv.imshow(f'original mask {name}, {category_name}', binarized_bool.astype("uint8") * 255)

    # remove small objects per escludere gi eventuali oggetti separati
    # e prima della resize per pulire l'immagine con più precisione

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