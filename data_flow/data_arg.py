import tensorflow as tf
import random
import numpy as np


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

    def arg_single(self, data, arg_dict, seed, is_list):
        """
            if seed == -1, the seed won't be set. For single data purpose
            if seed != -1, it will perform the same random for all
                    the data
        """

        # if seed != -1:
        #   tf.set_random_seed(seed)

        if "image_whiten" in arg_dict and arg_dict["image_whiten"]:
            data = tf.image.per_image_standardization(data)

        if "center_crop_frac" in arg_dict:
            data = tf.image.central_crop(data, arg_dict["center_crop_frac"])

        if "rbright_max" in arg_dict:
            data = tf.image.random_brightness(data,
                                              arg_dict["rbright_max"],
                                              seed=seed)

        if "rcontrast_lower_upper" in arg_dict:
            data = tf.image.random_contrast(data,
                                            arg_dict["rcontrast_lower_upper"][0],
                                            arg_dict["rcontrast_lower_upper"][1],
                                            seed=seed)

        if "rhue_max" in arg_dict:
            data = tf.image.random_hue(data,
                                       arg_dict["rhue_max"],
                                       seed=seed)

        if "rsat_lower_upper" in arg_dict:
            data = tf.image.random_saturation(data,
                                              arg_dict["rsat_lower_upper"][0],
                                              arg_dict["rsat_lower_upper"][1],
                                              seed=seed)

        if "ccrop_size" in arg_dict:
            i_height, i_width, i_cha = data.get_shape().as_list()
            ccrop_size = arg_dict["ccrop_size"]
            offset_height = int((i_height - ccrop_size[0]) / 2)
            offset_width = int((i_width - ccrop_size[1]) / 2)
            data = tf.image.crop_to_bounding_box(data,
                                                 offset_height, offset_width, ccrop_size[0], ccrop_size[1])

        if not is_list:
            if "rflip_updown" in arg_dict and arg_dict["rflip_updown"]:
                data = tf.image.random_flip_up_down(data, seed=seed)

            if "rflip_leftright" in arg_dict and arg_dict["rflip_leftright"]:
                data = tf.image.random_flip_left_right(data, seed)

            if "rcrop_size" in arg_dict:
                rcrop_size = arg_dict["rcrop_size"]
                data = tf.random_crop(data, rcrop_size, seed=seed)

        return data

    def center_padding(self, data, target_h, target_w):
        # h, w, c = data[0].get_shape()

        t_shape = tf.shape(data[0])
        h = t_shape[0]
        w = t_shape[1]

        offset_h = tf.cast(tf.subtract(target_h, h) / 2, tf.int32)
        offset_w = tf.cast(tf.subtract(target_w, w) / 2, tf.int32)
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
                        minval=arg_dict[i]["multiscale_range"][0],
                        maxval=arg_dict[i]["multiscale_range"][1],
                        dtype=tf.float32, seed=seed)
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
                                                     minval=0, maxval=offset_height_max,
                                                     dtype=tf.int32)

                        r_width = tf.random_uniform([],
                                                    minval=0, maxval=offset_width_max,
                                                    dtype=tf.int32)

                if offset_height_max == 0 and offset_width_max == 0:
                    pass
                else:
                    new_data[i] = tf.image.crop_to_bounding_box(new_data[i],
                                                                r_weight, r_width, rcrop_size[0], rcrop_size[1])
        return new_data

    def rflip_lr(self, data, arg_dict, seed):
        """for left right flip """
        rflip_lr_op = tf.random_uniform([], minval=0,
                                        maxval=2, dtype=tf.int32, seed=seed)
        mirror = tf.less(rflip_lr_op, 1)
        for i in range(len(data)):
            if "rflip_leftright" in arg_dict[i] and arg_dict[i]["rflip_leftright"]:
                data[i] = tf.cond(mirror, lambda: tf.reverse(
                    data[i], [False, True, False]), lambda: data[i])
        return data

    def rflip_ud(self, data, arg_dict, seed):
        rflip_ud_op = tf.random_uniform([], minval=0,
                                        maxval=2, dtype=tf.int32, seed=seed)
        mirror = tf.less(rflip_ud_op, 1)
        for i in range(len(data)):
            if "rflip_updown" in arg_dict[i] and arg_dict[i]["rflip_updown"]:
                data[i] = tf.cond(mirror, lambda: tf.reverse(
                    data[i], [0]), lambda: data[i])
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

    def rflip_lr_image_box(self, box, images):
        """ Random flip images and box.
        Args:
            box: [n, 4], [xmin, ymin, xmax, ymax]
            images: one or a list of tensors. each one has the same dims [h, w, c]

        """

        def flip_box(box, image_width):
            box_x_min = image_width - box[:, 2]
            box_x_max = image_width - box[:, 0]

            box = tf.stack([box_x_min, box[:, 1], box_x_max, box[:, 3]])

            if len(box.shape) == 1:
                box = tf.expand_dims(box, 0)
            else:
                box = tf.transpose(box)

            return box

        if not isinstance(images, list):
            images = [images]

        image_width = images[0].get_shape().as_list()[1]
        rflip_lr_op = tf.random_uniform([], minval=0,
                                        maxval=2, dtype=tf.int32)
        mirror = tf.less(rflip_lr_op, 1)
        box = tf.cond(mirror, lambda: flip_box(box, image_width), lambda: box)

        output_images = list()

        for image in images:
            output_images.append(
                tf.cond(mirror, lambda: tf.reverse(image, [1]), lambda: image))

        return box, output_images

    def rshift_image_box(self, box, images, height_shift_max, width_shift_max):
        """ Ramdon shift images and box.

        Args:
            box: [n, 4], [xmin, ymin, xmax, ymax]
            images: one or a list of tensors. each one has the same dims [h, w, c]
            shift_max: max shift amount.
        """
        is_list = True
        if not isinstance(images, list):
            is_list = False
            images = [images]

        image_height, image_width, _ = images[0].get_shape().as_list()

        assert height_shift_max < image_height
        assert width_shift_max < image_width

        shift_height = tf.random_uniform([], minval=-height_shift_max,
                                         maxval=height_shift_max, dtype=tf.float32)

        shift_width = tf.random_uniform([], minval=-width_shift_max,
                                        maxval=width_shift_max, dtype=tf.float32)

        def _py_shift_image(image, shift_delta, is_height):
            shift_delta = int(shift_delta)
            new_image = np.zeros(image.shape, image.dtype)
            if is_height:
                new_image[max(0, shift_delta): min(image_height, image_height + shift_delta), :] = \
                    image[max(0, -shift_delta): min(image_height,
                                                    image_height - shift_delta), :]
            else:
                new_image[:, max(0, shift_delta): min(image_width, image_width + shift_delta)] = \
                    image[:, max(0, -shift_delta): min(image_width,
                                                       image_width - shift_delta)]

            return new_image

        output_images = list()
        for image in images:
            new_image = tf.py_func(
                _py_shift_image, [image, shift_height, True], image.dtype)
            new_image = tf.py_func(
                _py_shift_image, [new_image, shift_width, False], image.dtype)
            new_image.set_shape(image.get_shape())
            output_images.append(new_image)

        box_x_min = box[:, 0] + shift_width
        box_x_max = box[:, 2] + shift_width
        box_y_min = box[:, 1] + shift_height
        box_y_max = box[:, 3] + shift_height

        box = tf.stack([box_x_min, box_y_min, box_x_max, box_y_max])
        if len(box.shape) == 1:
            box = tf.expand_dims(box, 0)
        else:
            box = tf.transpose(box)

        # If image is not list, then return image should not be list.
        if not is_list:
            output_images = output_images[0]

        return box, output_images
