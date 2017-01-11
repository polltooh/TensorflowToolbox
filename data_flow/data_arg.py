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
		if not isinstance(data, list):
			data = self.arg_single(data, arg_dict)
		else:
			seed = self.get_random_seed()
			for i in range(len(data)):
				data[i] = self.arg_single(data[i], arg_dict[i], seed)
		return data

	def get_random_seed(self):
		return random.randint(0, 20000)
		
	def arg_single(self,data, arg_dict, seed = -1):
		"""
			if seed == -1, the seed won't be set. For single data purpose
			if seed != -1, it will perform the same random for all
					the data
		"""

		if seed != -1:
			tf.set_random_seed(seed)

		if "image_whiten" in arg_dict and arg_dict["image_whiten"]:
			data = tf.image.per_image_standardization(data)

		if "center_crop_frac" in arg_dict:
			data = tf.image.central_crop(data, arg_dict["center_crop_frac"])


		if "rbright_max" in arg_dict:
			data = tf.image.random_brightness(data, 
					arg_dict["rbright+max"])

		if "rcontrast_lower_upper" in arg_dict:
			data = tf.image.random_contrast(data, 
					arg_dict["rcontrast_lower_upper"][0], 
					arg_dict["rcontrast_lower_upper"][1])

		if "rhue_max" in arg_dict:
			data = tf.image.random_hue(data, arg_dict["rhue_max"])

		if "rsat_lower_upper" in arg_dict:
			data = tf.image.random_saturation(data,
					arg_dict["rsat_lower_upper"][0], 
					arg_dict["rsat_lower_upper"][1])
		
		if "rflip_updown" in arg_dict and arg_dict["rflip_updown"]:
			data = tf.image.random_flip_up_down(data)

		if "rflip_leftright" in arg_dict and arg_dict["rflip_leftright"]:
			data = tf.image.random_flip_left_right(data)

		if "rcrop_size" in arg_dict:
			data = tf.random_crop(data, arg_dict["rcrop_size"])
			
		return data
			
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

