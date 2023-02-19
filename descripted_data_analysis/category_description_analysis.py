import numpy as np

#TODO far si che funzioni anche per i singoli descrittori
def category_dsc_analysis(descripted_data_file: str, descriptor: object):
    data= np.load(descripted_data_file, allow_pickle=True)
    samples = data[6]
    means, stds = data[7]
    descriptor.draw_tabular(samples, means, stds)
    descriptor.draw_samples(samples)
