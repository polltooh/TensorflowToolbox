from abc import ABCMeta, abstractmethod

class DataAbs(object):
    __metaclass__ = ABCMeta

    #@abstractmethod
    #def data_load(self, file_name):
    #   pass

    @abstractmethod
    def get_label(self):
        pass

    @abstractmethod
    def get_input(self):
        pass

