import abc
import json

import file_io


class FileLoader(metaclass=abc.ABCMeta):
    def __init__(self):
        self.curr_num = 0
        self.file_len = 0
        self.file_name = ''
        self.decoded_file = list()
        self.epoch = 0

    @abc.abstractmethod
    def read_file(self, file_name):
        pass

    @abc.abstractmethod
    def get_next(self, batch_size):
        pass


class TextFileLoader(FileLoader):
    """
    File format:

    e.g.
    image_name1.jpg 1
    image_name2.jpg 2
    """
    def __init__(self):
        super(TextFileLoader, self).__init__()

    def read_file(self, file_name, delimit=' '):
        self.file_name = file_name
        file_list = file_io.read_file(file_name)
        for f in file_list:
            f_list = f.split(delimit)
            self.decoded_file.append(f_list)
        self.file_len = len(self.decoded_file) 

    def get_next(self, batch_size):
        end_index = self.curr_num + batch_size
        if end_index >= self.file_len:
            end_index = self.file_len
            self.epoch += 1

        return_list = self.decoded_file[self.curr_num:end_index]

        if end_index == self.file_len:
            self.curr_num = 0
        else:
            self.curr_num = end_index

        return return_list


class JsonFileLoader(FileLoader):
    def __init__(self):
        super(JsonFileLoader, self).__init__()
   
    def read_file(self, file_name):
        self.file_name = file_name
        with open(file_name) as input_file:    
            self.decoded_file = json.load(input_file)

        self.file_len = len(self.decoded_file)
    
    def get_next(self, batch_size):
        end_index = self.curr_num + batch_size
        if end_index >= self.file_len:
            end_index = self.file_len
            self.epoch += 1
        
        return_list = self.decoded_file[self.curr_num:end_index]

        if end_index == self.file_len:
            self.curr_num = 0
        else:
            self.curr_num = end_index

        return return_list
