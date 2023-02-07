import sys

import numpy as np
from models import Models


def prepare_models(descriptors_file='data.npy', model_file='cl1.npy'):
    data = np.load(descriptors_file, allow_pickle=True)


    X = data[0]
    Y = data[1]
    file_names = data[2]
    category_legend = data[3]


    m = Models(X, Y, file_names, category_legend)
    m.train_and_test()
    #m.train()
    m.train_with_fold(fold=1)

    #Svuota la memoria del classificatore
    m.X = None
    m.labels = None

    # trained = np.array([m],dtype= object)

    trained = np.empty(1, dtype=object)

    trained[0] = m

    np.save(model_file, trained )
    print("BELLLLAAAAAAAAA")


if __name__ == '__main__':
    print("esecuzione di train.py")
    prepare_models()
    sys.exit()