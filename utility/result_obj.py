import file_io

class ResultObj(object):
    def __init__(self, result_file_name):
        self.result_file_name = result_file_name
        self.file_list = list()

    def add_to_list(self, file_name_list, *args):
        curr_list = list()
        for i, f in enumerate(file_name_list):
            print(f)
            curr_str = f + " " 
            for arg in args:
                curr_str = curr_str + arg[i] + " "

            # remove last empty space
            curr_str = curr_str[:-1] 
            curr_list.append(curr_str)

        self.file_list += curr_list

    def save_to_file(self, sort_result = True):
        if sort_result:
            self.file_list.sort()
        file_io.save_file(self.file_list, self.result_file_name)

    def float_to_str(self, float_list, sformat):
        str_list = list()
        for f in float_list:
            str_list.append(sformat%(f))

        return str_list


