import abc
import json
import random
import threading

import file_io


class FileLoader():
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self.curr_num = 0
        self.file_len = 0
        self.file_name = ''
        self.decoded_file = list()
        self.epoch = 0
        self.lock = threading.Lock()

    def shuffle_data(self, file_list):
        random.shuffle(file_list)

    def get_file_len(self):
        return self.file_len

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

    def read_file(self, file_name, delimit=' ', shuffle=False):
        self.file_name = file_name
        file_list = file_io.read_file(file_name)
        if shuffle:
            self.shuffle_data(file_list)

        for f in file_list:
            f_list = f.split(delimit)
            self.decoded_file.append(f_list)
        self.file_len = len(self.decoded_file)

    def get_next(self, batch_size):
        assert batch_size == 1
        with self.lock:
            curr_num = self.curr_num
            end_index = self.curr_num + batch_size
            if end_index >= self.file_len:
                end_index = self.file_len
                self.epoch += 1
                if shuffle:
                    self.shuffle_data(file_list)

            if end_index == self.file_len:
                self.curr_num = 0
            else:
                self.curr_num = end_index

        return_list = self.decoded_file[curr_num:end_index]
        return return_list


class JsonFileLoader(FileLoader):
    def __init__(self):
        super(JsonFileLoader, self).__init__()

    def read_file(self, file_name, shuffle=False):
        self.file_name = file_name
        with open(file_name) as input_file:
            self.decoded_file = json.load(input_file)
        if shuffle:
            self.shuffle_data(decoded_file)

        self.file_len = len(self.decoded_file)

    def get_next(self, batch_size):
        assert batch_size == 1
        with self.lock:
            curr_num = self.curr_num
            end_index = self.curr_num + batch_size
            if end_index >= self.file_len:
                end_index = self.file_len
                self.epoch += 1
                if shuffle:
                    self.shuffle_data(decoded_file)

            if end_index == self.file_len:
                self.curr_num = 0
            else:
                self.curr_num = end_index

        return_list = self.decoded_file[curr_num:end_index]

        return return_list
