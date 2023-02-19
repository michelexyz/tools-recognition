import abc
import time

import numpy as np


class DescriptorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'describe') and
                callable(subclass.describe) and
                hasattr(subclass, 'get_dim') and
                callable(subclass.get_dim) or
                NotImplemented)

    @abc.abstractmethod
    def describe(self, componentMask, componentMaskBool, name, draw):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_dim(self):
        """Extract text from the data set"""
        raise NotImplementedError



class DescriptorCombiner(DescriptorInterface):
    def __init__(self, descriptors: list, weights:list = None):
        self.descriptors = descriptors
        self.dim, self.dims = self.compute_dim()
        self.weights = weights
        if self.weights is None:
            self.weights = np.ones_like(descriptors).tolist()
        if len(weights) != len(descriptors):
            raise ("number of weights given doesn't match number of descriptors")

    def describe(self, componentMask, componentMaskBool, name, draw = True):
        """Overrides DescriptorInterface.describe()"""
        descriptions = []
        for dsc in self.descriptors:
            descriptions.append(dsc.describe(componentMask, componentMaskBool, name=name, draw=draw))

        return np.concatenate(descriptions)

    def get_dim(self):
        """Overrides DescriptorInterface.get_dim()"""

        return self.dim, self.dims

    #TODO far si che i draw funzionino anche per i singoli descrittori
    def draw_tabular(self, samples, means=None, stds=None):
        for i, dsc in enumerate(self.descriptors):
            print(f'Tabelle descrittore {dsc.__class__.__name__}')
            dsc.draw_tabular(samples, means[i], stds[i])
            # Waiting for pycharm plotting delay
            print("WAITING 3 SECONDS")
            time.sleep(3)

    def draw_samples(self, samples):
        for dsc in self.descriptors:
            print(f'Disegno sample per descrittore {dsc.__class__.__name__}')
            dsc.draw_samples(samples)
            # Waiting for pycharm plotting delay
            print("WAITING 3 SECONDS")
            time.sleep(3)

    def compute_dim(self):
        dim = 0
        dims = []
        for dsc in self.descriptors:
            dsc_dim,_ = dsc.get_dim()
            dim += dsc_dim
            dims.append(dsc_dim)
        return dim, dims






