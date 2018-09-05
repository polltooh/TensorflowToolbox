import json


class Dataset(object):
    def __init__(self, json_file_name, parser_fn):
        self.data_list = json.load(open(file_name, 'r'))
        if not isinstance(self.data_list, list):
            raise ValueError("Expect data list is a list")
        self._parser_fn = parser_fn

    def next(self):
        for data in self._data_list:
            yield self._parser_fn(data)
