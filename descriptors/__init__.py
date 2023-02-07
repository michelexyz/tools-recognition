import abc


class DescriptorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'describe') and
                callable(subclass.describe) and
                hasattr(subclass, 'get_dim') and
                callable(subclass.get_dim) or
                NotImplemented)

    @abc.abstractmethod
    def describe(self, componentMask, componentMaskBool, area):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_dim(self, full_file_path: str):
        """Extract text from the data set"""
        raise NotImplementedError





