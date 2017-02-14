import file_io

def check_bool(val):
    if val == 'true' or val == "True":
        val = True
    if val == 'false' or val == "False":
        val = False
    return val

def check_digit(val):
    if val.replace(".","").isdigit():
        val = float(val)
        if val.is_integer():
            val = int(val)
    return val  

def check_list(val):
    if isinstance(val, str):
        if len(val) > 1 and val[0] == '[':
            val = val[1:-1]
            val = val.split(',')
            for i in range(len(val)):
                val[i] = check_bool(val[i])
                val[i] = check_digit(val[i])
    return val

def check_none(val):
    if isinstance(val, str):
        if val == "None" or val == "none":
            return None
    return val

def load_proto(file_name):
    model_param = dict()
    param = file_io.read_file(file_name)
    for li in param:
        li = li.replace(" ", "")
        if len(li) == 0 or li[0] == "#":
            continue
        name, val = li.split(":")

        val = check_digit(val)
        val = check_bool(val)
        val = check_list(val)
        val = check_none(val)

        model_param[name] = val

    return model_param

if __name__ == "__main__":
    file_name = "../model.tfproto"
    model_param = load_proto(file_name)
    print(model_param)
    print(type(model_param["data_list"]))
