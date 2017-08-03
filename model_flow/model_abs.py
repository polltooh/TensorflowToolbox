from abc import ABCMeta, abstractmethod
import tensorflow as tf

class ModelAbs(ABCMeta):
    # __metaclass__ = ABCMeta

    @abstractmethod
    def model_infer(self, input_data, is_train):
        raise NotImplementedError

    @abstractmethod
    def model_loss(self, input_data, output):
        raise NotImplementedError

    @abstractmethod
    def model_optimizer(self):
        raise NotImplementedError
