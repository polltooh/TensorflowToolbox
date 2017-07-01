import abc


class InputLayerTemp(abc.ABCMeta):

    @abc.abstractmethod
    def read_data(self):
        pass

    @abc.abstractmethod
    def process_data(self):
        pass

    
