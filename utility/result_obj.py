import file_io
import numpy as np

class ResultObj(object):
    def __init__(self, result_file_name, is_double_list = False):
        self.result_file_name = result_file_name
        self.file_list = list()

        # in case of lstm, the file_name_list
        # will be a double list
        self.is_double_list = is_double_list 

    def add_to_list(self, file_name_list, *args):
        curr_list = list()
        for i, f in enumerate(file_name_list):
            curr_str = f + " " 
            for arg in args:
                curr_str = curr_str + arg[i] + " "

            # remove last empty space
            curr_str = curr_str[:-1] 
            curr_list.append(curr_str)

        self.file_list += curr_list

    def check_if_double_list(self, *args):
        args = list(args)
        if not isinstance(args[0][0], list):
            return args

        self.is_double_list = True
        for i, arg in enumerate(args):
            args[i] = self.vectorize_list(arg)
        return args

    def vectorize_nparray(self, double_np_array):
        """
        Args:
            double_np_array: list of np array in the 
            case of lstm

        Return:
            concatenated np array
        """
        np_array = np.array(double_np_array)
        np_array = np.transpose(np_array)
        np_array = np.reshape(np_array, (-1))

        return np_array

    def vectorize_list(self, double_list):
        new_list = list()
        for l in double_list:
            new_list += l
        return new_list


    def save_to_file(self, sort_result = True):
        if sort_result:
            self.file_list.sort()
        file_io.save_file(self.file_list, self.result_file_name)

    def float_to_str(self, float_list, sformat):
        str_list = list()
        for f in float_list:
            str_list.append(sformat%(f))

        return str_list

