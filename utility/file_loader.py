import abc
import json
import random
import threading

import numpy as np

import file_io


class FileLoader():
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self._curr_index = -1
        self._file_len = 0
        self._file_name = ''
        self._decoded_file = list()
        self._epoch = 0
        self._shuffle = False
        self._file_index_array = None

        self.lock = threading.Lock()

    def _shuffle_index(self):
        if self._shuffle:
            random.shuffle(self._file_index_array)

    def _get_next_index(self):
        next_index = self._curr_index + 1
        if next_index == self._file_len:
            self._shuffle_index()
            self._epoch += 1
            next_index = 0
        self._curr_index = next_index
        return self._file_index_array[next_index]

    def _pre_read_file(self, file_name, shuffle):
        self._file_name = file_name
        self._shuffle = shuffle


    def _post_read_file(self):
        self._file_len = len(self._decoded_file)
        self._file_index_array = range(self._file_len)
        self._shuffle_index()

    def get_file_len(self):
        return self._file_len

    def get_next(self):
        with self.lock:
            next_index = self._get_next_index()

        next_item  = self._decoded_file[next_index]
        return next_item

    @abc.abstractmethod
    def read_file(self, file_name):
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

    def read_file(self, file_name, shuffle=False, delimit=' '):
        self._pre_read_file(file_name, shuffle)

        file_list = file_io.read_file(file_name)
        for f in file_list:
            f_list = f.split(delimit)
            self._decoded_file.append(f_list)

        self._post_read_file()


class JsonFileLoader(FileLoader):
    """
    File format:

    e.g.
        [{key1: 1}, {key2: 1}]
        [{key1: 2}, {key2: 2}]
        ...
    
    """
    def __init__(self):
        super(JsonFileLoader, self).__init__()

    def read_file(self, file_name, shuffle=False):
        self._pre_read_file(file_name, shuffle)

        with open(file_name) as input_file:
            self._decoded_file = json.load(input_file)

        self._post_read_file()
