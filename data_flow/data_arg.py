import tensorflow as tf
import random

class DataArg(object):
    def __init__(self):
        """ the image should be float32 type [0, 1]
        
        arg_dict["rbright_max"] = 0.2# [0,1]
        arg_dict["rcontrast_lower"] = 0.5 #[0,1]
        arg_dict["rcontrast_upper"] = 1.5 #[1,2]
        arg_dict["rhue_max"] = 0.2 #[0, 0.5]
        arg_dict["rsat_lower"] = 0.5 #[0,1]
        arg_dict["rsat_upper"] = 1.5 #[1,2]
        arg_dict["image_whiten"] = False #[False or True]
        arg_dict["rflip_updown"] = False #[False or True]
        arg_dict["rflip_leftright"] = False #[False or True]
        arg_dict["center_crop_frac"] = 0.2 #[0,1]

        arg_dict["rcrop_size"] = [256,256] 
                    #random crop to size [crop_height, crop_width, crop_channel]

        arg_dict["ccrop_size"] = [256,256] 
                    # center crop to size [crop_height, crop_width, crop_channel]
        
        """
        pass

    def __call__(self, data, arg_dict):
        seed = self.get_random_seed()
        if not isinstance(data, list):
            data = self.arg_single(data, arg_dict, seed, False)
        else:
            data = self.arg_list(data, arg_dict, seed)
            for i in range(len(data)):
                data[i] = self.arg_single(data[i], arg_dict[i], seed, True)

        return data

    def get_random_seed(self):
        return random.randint(0, 20000)
    
    def get_random_bboffset(self, wmax, hmax, seed):
        random.seed(seed)
        woffset = random.randint(0, wmax)
        hoffset = random.randint(0, hmax)
        return woffset, hoffset
        
    def arg_single(self,data, arg_dict, seed, is_list):
        """
            if seed == -1, the seed won't be set. For single data purpose
            if seed != -1, it will perform the same random for all
                    the data
        """

        #if seed != -1:
        #   tf.set_random_seed(seed)

        if "image_whiten" in arg_dict and arg_dict["image_whiten"]:
            data = tf.image.per_image_standardization(data)

        if "center_crop_frac" in arg_dict:
            data = tf.image.central_crop(data, arg_dict["center_crop_frac"])

        if "rbright_max" in arg_dict:
            data = tf.image.random_brightness(data, 
                    arg_dict["rbright_max"],
                    seed = seed)

        if "rcontrast_lower_upper" in arg_dict:
            data = tf.image.random_contrast(data, 
                    arg_dict["rcontrast_lower_upper"][0], 
                    arg_dict["rcontrast_lower_upper"][1],
                    seed = seed)

        if "rhue_max" in arg_dict:
            data = tf.image.random_hue(data, 
                    arg_dict["rhue_max"],
                    seed = seed)

        if "rsat_lower_upper" in arg_dict:
            data = tf.image.random_saturation(data,
                    arg_dict["rsat_lower_upper"][0], 
                    arg_dict["rsat_lower_upper"][1],
                    seed = seed)

        if "ccrop_size" in arg_dict:
            i_height, i_width, i_cha = data.get_shape().as_list()
            ccrop_size = arg_dict["ccrop_size"]
            offset_height = int((i_height - ccrop_size[0])/2)
            offset_width = int((i_width - ccrop_size[1])/2)
            data = tf.image.crop_to_bounding_box(data, 
                    offset_height, offset_width, ccrop_size[0], ccrop_size[1])

        if not is_list:
            if "rflip_updown" in arg_dict and arg_dict["rflip_updown"]:
                data = tf.image.random_flip_up_down(data, seed = seed)

            if "rflip_leftright" in arg_dict and arg_dict["rflip_leftright"]:
                data = tf.image.random_flip_left_right(data, seed)

            if "rcrop_size" in arg_dict:
                rcrop_size = arg_dict["rcrop_size"]
                data = tf.random_crop(data, rcrop_size, seed = seed)


        return data

    def center_padding(self, data, target_h, target_w):
        # h, w, c = data[0].get_shape()

        t_shape = tf.shape(data[0])
        h = t_shape[0]
        w = t_shape[1]

        offset_h = tf.cast(tf.subtract(target_h, h)/2, tf.int32)
        offset_w = tf.cast(tf.subtract(target_w, w)/2, tf.int32)
        new_data = [tf.identity(d) for d in data]
        for i in range(len(new_data)): 
            new_data[i] = tf.image.pad_to_bounding_box(
                        new_data[i],
                        offset_h,
                        offset_w,
                        target_h,
                        target_w)

        return new_data 

    def rmultiscale(self, data, arg_dict, seed):
        org_h, org_w, org_c = data[0].get_shape().as_list()
        activate_multi = False
        for i in range(len(data)):
            if "multiscale_range" in arg_dict[i]:
                if not activate_multi:
                    activate_multi = True
                    rscale_op = tf.random_uniform(
                                        [],
                                        minval = arg_dict[i]["multiscale_range"][0],
                                        maxval = arg_dict[i]["multiscale_range"][1],
                                        dtype = tf.float32, seed = seed)
                h, w, c = data[i].get_shape().as_list()
                if "multiscale_resize" in arg_dict[i]:
                    if arg_dict[i]["multiscale_resize"] is "BILINEAR":
                        method = tf.image.ResizeMethod.BILINEAR
                    elif arg_dict[i]["multiscale_resize"] is "NN":
                        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                    else:
                        raise NotImplementedError
                else:
                    method = tf.image.ResizeMethod.BILINEAR
                data[i] = tf.image.resize_images(
                                    data[i],
                                    tf.cast([h, w] * rscale_op, tf.int32), 
                                    method)

        if activate_multi:
            data = tf.cond(tf.less_equal(tf.cast(h * rscale_op, tf.int32), org_h),
                       lambda: self.center_padding(data, org_h, org_w),
                       lambda: self.rcrop(data, [{'rcrop_size': [org_h, org_w]}] * len(data), 
                               seed))
        return data

    def rcrop(self, data, arg_dict, seed):
        """ for random crop """
        activate_rcrop = False
        new_data = [tf.identity(d) for d in data]
        for i in range(len(new_data)):
            if "rcrop_size" in arg_dict[i]:
                if not activate_rcrop:
                    activate_rcrop = True
                    # i_height, i_width, i_cha = data[i].get_shape().as_list()
                    data_shape = tf.shape(new_data[i])
                    i_height = data_shape[0]
                    i_width = data_shape[1]
                    i_cha = data_shape[2]
                    rcrop_size = arg_dict[i]["rcrop_size"]
                    offset_height_max = i_height - rcrop_size[0]
                    offset_width_max = i_width - rcrop_size[1]

                    if offset_height_max == 0 and offset_width_max == 0:
                        pass
                    else:
                        r_weight = tf.random_uniform([],
                                    minval = 0, maxval = offset_height_max, 
                                    dtype=tf.int32) 

                        r_width = tf.random_uniform([], 
                                    minval = 0, maxval = offset_width_max, 
                                    dtype=tf.int32) 

                if offset_height_max == 0 and offset_width_max == 0:
                    pass
                else:
                    new_data[i] = tf.image.crop_to_bounding_box(new_data[i], 
                            r_weight, r_width, rcrop_size[0], rcrop_size[1])
        return new_data
        
    def rflip_lr(self, data, arg_dict, seed):
        """for left right flip """
        rflip_lr_op = tf.random_uniform([], minval = 0,
                    maxval = 2, dtype = tf.int32, seed = seed)
        mirror = tf.less(rflip_lr_op, 1)
        for i in range(len(data)):
            if "rflip_leftright" in arg_dict[i] and arg_dict[i]["rflip_leftright"]:
                data[i] = tf.cond(mirror, lambda: tf.reverse(data[i], [1]), lambda: data[i])
        return data
    
    def rflip_ud(self, data, arg_dict, seed):
        rflip_ud_op = tf.random_uniform([], minval = 0, 
                    maxval = 2, dtype = tf.int32, seed = seed)
        mirror = tf.less(rflip_ud_op, 1)
        for i in range(len(data)):
            if "rflip_updown" in arg_dict[i] and arg_dict[i]["rflip_updown"]:
                data[i] = tf.cond(mirror, lambda: tf.reverse(data[i], [0]), lambda: data[i])
        return data
        

    def arg_list(self, data, arg_dict, seed):
        assert(isinstance(data, list))
        assert(len(data) == len(arg_dict))

        """for up down flip """
        data = self.rflip_ud(data, arg_dict, seed)

        """for left right flip """
        data = self.rflip_lr(data, arg_dict, seed)

        """ for multi scale """
        data = self.rmultiscale(data, arg_dict, seed)

        """ for random crop """
        data = self.rcrop(data, arg_dict, seed)

        return data
