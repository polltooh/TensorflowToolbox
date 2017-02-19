from abc import ABCMeta, abstractmethod
import tensorflow as tf

class ModelAbs(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def model_infer(self, data_ph, model_params):
        raise NotImplementedError

    @abstractmethod
    def model_loss(self, data_ph, model_params):
        raise NotImplementedError

    @abstractmethod
    def model_mini(self, loss):
        raise NotImplementedError
    
