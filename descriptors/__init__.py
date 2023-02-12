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
    def get_dim(self, full_file_path: str):
        """Extract text from the data set"""
        raise NotImplementedError

class DescriptorCombiner(DescriptorInterface):
    def __init__(self, descriptors: list):
        self.descriptors = descriptors
        self.dim = self.compute_dim()
    def describe(self, componentMask, componentMaskBool, area, name, draw = True):
        """Overrides DescriptorInterface.describe()"""
        descriptions = []
        for dsc in self.descriptors:
            descriptions.append(dsc.describe(componentMask, componentMaskBool, area, name= name, draw=draw))

        return np.concatenate(descriptions)

    def get_dim(self):
        """Overrides DescriptorInterface.get_dim()"""

        return self.dim

    def compute_dim(self):
        dim = 0
        for dsc in self.descriptors:
            dim += dsc.get_dim()
        return dim






