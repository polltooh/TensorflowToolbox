import file_io

def load_proto(file_name):
    model_param = dict()
    param = file_io.read_file(file_name)
    for li in param:
        li = li.replace(" ", "")
        name, val = li.split(":")
        if val.replace(".","").isdigit():
            val = float(val)
            if val.is_integer():
                val = int(val)
        if val == 'true' or val == "True":
            val = True
        if val == 'false' or val == "False":
            val = False

        model_param[name] = val
    return model_param

if __name__ == "__main__":
	file_name = "model_proto"
	model_param = load_proto(file_name)
