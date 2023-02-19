import numpy as np
def load_mean_and_std(described_data_file, descriptor):
    data = np.load(described_data_file, allow_pickle=True)
    X = data[0]
    Y = data[1]
    means, stds = compute_mean_and_std(X,Y, descriptor)
    data[7] = (means, stds)
    np.save(described_data_file, data)


def compute_mean_and_std(X, Y, descriptor, n_categories = None):
    categories = np.unique(Y)
    dim, dims = descriptor.get_dim()
    means = np.empty((len(dims), categories.shape[0] ), dtype=np.ndarray)
    stds = np.empty((len(dims), categories.shape[0] ), dtype=np.ndarray)
    # if n_categories is None:
    #     n_categories = np.unique(Y).shape[0]

    for c_index, c in enumerate(categories):
        category_mask = Y == c
        category_x = X[category_mask]
        start = 0
        for d_index, d in enumerate(dims):
            descriptor_x = category_x[:, start: start + d]
            means[d_index,c_index] = np.mean(descriptor_x,axis=0)
            stds[d_index,c_index] = np.std(descriptor_x,axis=0)
            start += d


    return means, stds








