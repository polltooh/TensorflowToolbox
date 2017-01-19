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
		arg_dict["rcrop_size"] = [256,256,3] #[crop_height, crop_width, crop_channel]
		
		"""
		pass

	def __call__(self, data, arg_dict):
		seed = self.get_random_seed()
		if not isinstance(data, list):
			data = self.arg_single(data, arg_dict, seed, False)
		else:
			for i in range(len(data)):
				data[i] = self.arg_single(data[i], arg_dict, seed, True)
			data = self.arg_list(data, arg_dict, seed)

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
		#	tf.set_random_seed(seed)

		if "image_whiten" in arg_dict and arg_dict["image_whiten"]:
			data = tf.image.per_image_standardization(data)

		if "center_crop_frac" in arg_dict:
			data = tf.image.central_crop(data, arg_dict["center_crop_frac"])

		if "rbright_max" in arg_dict:
			data = tf.image.random_brightness(data, 
					arg_dict["rbright+max"],
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

		if not is_list:
			if "rflip_updown" in arg_dict and arg_dict["rflip_updown"]:
				data = tf.image.random_flip_up_down(data, seed = seed)

			if "rflip_leftright" in arg_dict and arg_dict["rflip_leftright"]:
				data = tf.image.random_flip_left_right(data, seed)

			if "rcrop_size" in arg_dict:
				rcrop_size = arg_dict["rcrop_size"]
				data = tf.random_crop(data, rcrop_size, seed = seed)

		return data

	def arg_list(self, data, arg_dict, seed):
		assert(isinstance(data, list))

		if "rflip_updown" in arg_dict and arg_dict["rflip_updown"]:
			rflip_ud_op = tf.random_uniform([], minval = 0, 
						maxval = 2, dtype = tf.int32, seed = seed)
			mirror = tf.less(tf.pack([rflip_ud_op, 2, 2]), 1)
			for i in range(len(data)):
				data[i] = tf.reverse(data[i], mirror)

		if "rflip_leftright" in arg_dict and arg_dict["rflip_leftright"]:
			rflip_lr_op = tf.random_uniform([], minval = 0, 
						maxval = 2, dtype = tf.int32, seed = seed)
			mirror = tf.less(tf.pack([2, rflip_lr_op, 2]), 1)
			for i in range(len(data)):
				data[i] = tf.reverse(data[i], mirror)

		if "rcrop_size" in arg_dict:
			i_height, i_width, i_cha = data[0].get_shape().as_list()
			rcrop_size = arg_dict["rcrop_size"]
			offset_height_max = i_height - rcrop_size[0]
			offset_width_max = i_width - rcrop_size[1]

			r_weight = tf.random_uniform([], 
						minval = 0, maxval = offset_height_max, 
						dtype=tf.int32)	

			r_width = tf.random_uniform([], 
						minval = 0, maxval = offset_width_max, 
						dtype=tf.int32)

			for i in range(len(data)):
				data[i] = tf.image.crop_to_bounding_box(data[i], 
						r_weight, r_width, rcrop_size[0], rcrop_size[1])

		return data

			#if rcrop_size[2] == 1:
			#	rcrop_size[2] = 3
			#	data = tf.tile(data, [1,1,3])
			#	data = tf.random_crop(data, 
			#		rcrop_size, seed = 0)
			#	data = data[:,:,1]
			#	data = tf.expand_dims(data, 2)
			#else:
			#	data = tf.random_crop(data, 
			#		rcrop_size, seed = 0)
			#data_w = data.get_shape().as_list()[0]
			#data_h = data.get_shape().as_list()[1]
			#woffset, hoffset = self.get_random_bboffset(
			#			data_w - rcrop_size[0], 
			#			data_h - rcrop_size[1], seed)

			#data = tf.image.crop_to_bounding_box(data, 
			#			woffset, hoffset, rcrop_size[0], rcrop_size[1])


			
			
		#arg_dict.image_whiten = False [False or True]
		#arg_dict.center_crop_frac = 0.5 [0,1]

		#arg_dict.rbright_max = 0.2 [0,1]
		#arg_dict.rcontrast_lower_upper = [0.5, 1.5] [0,1] [1,2]
		#arg_dict.rhue_max = 0.2 [0, 0.5]
		#arg_dict.rsat_lower_upper = [0.5, 1.5] [0,1] [1,2]
		#arg_dict.rflip_updown = False [False or True]
		#arg_dict.rflip_leftright = False [False or True]
		#arg_dict.rcrop_size = [256,256,3] [crop_height, crop_width, crop_channel]

		#tf.image.random_brightness(image, max_delta, seed=None)
		#tf.image.random_contrast(image, lower, upper, seed=None)
		#tf.image.random_hue(image, max_delta, seed=None)
		#tf.image.random_saturation(image, lower, upper, seed=None)
		#tf.image.random_flip_up_down(image, seed=None)
		#tf.image.random_flip_left_right(image, seed=None)
		#tf.random_crop(value, size, seed=None, name=None)

