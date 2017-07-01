tf2np = {
'tf.float32':'np.float32',
'tf.float64':'np.float64',
'tf.int16':'np.int16',
'tf.int32':'np.int32',
'tf.int64':'np.int64',
'tf.uint8':'np.uint8',
'tf.uint16':'np.uint16',
'tf.string':None,
'tf.bool':'np.bool',
'tf.complex64':None,
'tf.complex128':None,
'tf.qint8':None,
'tf.qint32':None,
'tf.quint8':None
}

def switch_key_val(input_dict):
    output_dict = dict()
    for key, value in input_dict.items():
        if value is not None:
            output_dict[value] = key

    return output_dict

np2tf = switch_key_val(tf2np)
print(np2tf)
print(tf2np)
