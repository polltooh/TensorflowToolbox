import argparse

from TensorflowToolbox.utility import file_io

import param_list

"""
Note:
    the key in the txt file has to be end with ':'. Otherwise it
    won't be able to parse the file and find the position.
"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--param_list', type=str, required=True)

    args = parser.parse_args()

    param_list = file_io.import_module_class(args.param_list.replace(".py", ""))

    key = param_list.key
    value_list = param_list.value_list
    save_name_ext = param_list.save_name_ext

    file_name = args.file_name

    with open(args.file_name, 'r') as f:
        file_string = f.read()

    index = 0
    while(1):
        index = file_string.find(key, index)
        if file_string[index+len(key)] == ":":
            break

    colon_position = index + len(key)
    for i, value in enumerate(value_list):
        end_position = file_string.find('\n', colon_position)
        new_file_s = file_string.replace(file_string[colon_position+1:end_position], str(value))
        save_name = ''.join([file_name, '.', save_name_ext[i]])
        with open(save_name, 'w') as f:
            f.write(new_file_s)

