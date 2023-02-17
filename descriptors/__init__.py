import abc

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
    def describe(self, componentMask, componentMaskBool, area, name, draw):
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

    def describe(self, componentMask, componentMaskBool, area, name, draw = True):
        """Overrides DescriptorInterface.describe()"""
        descriptions = []
        for dsc in self.descriptors:
            descriptions.append(dsc.describe(componentMask, componentMaskBool, area, name= name, draw=draw))

        return np.concatenate(descriptions)

    def get_dim(self):
        """Overrides DescriptorInterface.get_dim()"""

        return self.dim, self.dims

    def compute_dim(self):
        dim = 0
        dims = []
        for dsc in self.descriptors:
            dsc_dim,_ = dsc.get_dim()
            dim += dsc_dim
            dims.append(dsc_dim)
        return dim, dims






