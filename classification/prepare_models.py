import sys

import numpy as np
from classification.models import Models

from files.files_handler import get_file_abs_path


def prepare_models(descriptors_file='data.npy', output_file='cl1.npy'):
    data = np.load(descriptors_file, allow_pickle=True)


    X = data[0]
    Y = data[1]
    file_names = data[2]
    category_legend = data[3]

    #cat = category_legend[0].reshape(-1)
    m = Models(X, Y, file_names, category_legend[0].reshape(-1))
    m.train_and_test()
    m.train()
    #m.train_with_fold(fold=1)

    #Svuota la memoria del classificatore
    m.X = None
    m.labels = None

    # trained = np.array([m],dtype= object)

    trained = np.empty(1, dtype=object)

    trained[0] = m

    np.save(output_file, trained)


if __name__ == '__main__':
    print("esecuzione di prepare_models.py")

    desc_file = str(get_file_abs_path('data.npy'))
    output_file = str(get_file_abs_path('cl1.npy'))

    prepare_models(desc_file, output_file)
    sys.exit()