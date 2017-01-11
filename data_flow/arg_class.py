
class ArgClass(object):
	def __init__(self): 
		pass

	def __call__(self, b):
		a = b
		return a

	def default_params(self):
		self.rbright_max = 0.2# [0,1]
		self.rcontrast_lower = 0.5 #[0,1]
		self.rcontrast_upper = 1.5 #[1,2]
		self.rhue_max = 0.2 #[0, 0.5]
		self.rsat_lower = 0.5 #[0,1]
		self.rsat_upper = 1.5 #[1,2]
		self.image_whiten = False #[False or True]
		self.rflip_updown = False #[False or True]
		self.rflip_leftright = False #[False or True]
		self.center_crop = False #[False or True]
		self.rcorp_size = [256,256,3] #[crop_height, crop_width, crop_channel]
	
if __name__ == "__main__":
	a = ArgClass()

	print(a(2))
