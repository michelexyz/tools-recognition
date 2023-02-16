import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_data(input_file, scaled_file, scale_file, weights = None):

    data = np.load(input_file, allow_pickle=True)
    scaler = StandardScaler()
    X = data[0]
    Y = data[1]
    scaler.fit(X)
    X = scaler.transform(X)

    file_names = data[2]
    category_legend = data[3][0]
    print(category_legend)

    dims = data[5]
    n_descriptors = dims.shape[0]

    if weights is None :
        weights= np.ones(n_descriptors).tolist()

    if len(weights) != n_descriptors:
        raise("number of weights given doesn't match number of descriptors" )

    #scalers = np.empty(n_descriptors, dtype=object)

    # start = 0
    # for i, dim in enumerate(dims):
    #     scaler = StandardScaler()
    #     X[:, start: start + dim] = scaler.fit_transform(X[:, start: start + dim])*weights[i]
    #     scalers[i] = scaler
    #     start += dim

    start = 0
    for i, dim in enumerate(dims):
        X[:, start: start + dim] = X[:, start: start + dim] * weights[i]

        start += dim

    data[0] = X

    np.save(scaled_file, data)

    scaling_data = np.empty(1, dtype=object)
    scaling_data[0] = (dims, scaler, weights)
    #cale_data[1] = weights
    np.save(scale_file, scaling_data)