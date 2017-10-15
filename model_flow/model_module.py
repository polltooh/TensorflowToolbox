import tensorflow as tf

import model_func as mf

def conv_bottle_net(
            inputs, 
            bottlenet_channel, 
            filters, 
            data_format, 
            is_train, 
            leaky_param, 
            wd):

    bottlenet = mf.convolution_2d_layer(inputs, bottlenet_channel, [1, 1], [1, 1], "SAME",
                                        data_format, True, is_train, leaky_param, wd, 'bottlenet')

    conv = mf.convolution_2d_layer(bottlenet, filters, [3, 3], [1, 1], "SAME",
                                     data_format, True, is_train, leaky_param, wd, 'conv')
    return conv


def dense_conv_bc_block(
                input_tensor, 
                data_format, 
                is_train, 
                leaky_param, 
                wd, 
                growth_rate, 
                block_num,
                layer_name):
    """
    Args:
        layer_num: number of block. 2 layers.
    """
    if data_format == "NCHW":
        dense_concat_dim = 1
    elif data_format == "NHWC":
        dense_concat_dim = 3
    else:
        raise NotImplementedError

    conv = mf.dense_layer(input_tensor, block_num, dense_concat_dim, layer_name,
                conv_bottle_net, 3 * growth_rate, growth_rate, data_format, 
                is_train, leaky_param, wd)
    return conv


def dense_transition_layer(input_tensor, is_pool, data_format, is_train,
                           leaky_param, wd, layer_name):
    with tf.variable_scope(layer_name):
        if data_format == "NCHW":
            in_channle = input_tensor.get_shape()[1]
        elif data_format == "NHWC":
            in_channle = input_tensor.get_shape()[3]
        else:
            raise NotImplementedError

        if not is_pool:
            conv = mf.convolution_2d_layer(input_tensor, in_channle, [3, 3], [2, 2], "SAME",
                    data_format, True, is_train, leaky_param, wd, "conv")
        else:
            conv = mf.convolution_2d_layer(input_tensor, in_channle, [1, 1], [1, 1], "SAME",
                    data_format, True, is_train, leaky_param, wd, "conv")
            conv = mf.maxpool_2d_layer(conv, [2, 2], [2, 2], data_format, 'maxpool')
        return conv
