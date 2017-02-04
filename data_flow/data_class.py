import tensorflow as tf
import copy

def create_list_object(Object, count):
    """ 
    create a list of obejct using deep copy in 
    cased used in different theads

    Args:
            Object: object to be copied
            count: the number of copies
    Return:
            a list of objects
    """
    res_list = []
    for _ in xrange(count):
            res_list.append(Object)
    return res_list	

class DataClass():
    """ DataClass:
            used for decode line
    """
    def __init__(self, data_format):
        self.data_format = data_format
        self.decode_class = None

class BINClass():
    """ 
            used for load binary file
    """
    def __init__(self, shape, dtype = tf.float32):
        """ shape: a list """
        self.decode_fun = tf.decode_raw	
        self.dtype = dtype
        self.shape = shape

    def decode(self, filename):
        """ distort_data and whiten_data are not used """
        bin_file = tf.read_file(filename)
        bin_tensor = tf.decode_raw(bin_file, self.dtype)
        bin_tensor = tf.to_float(bin_tensor)
        bin_tensor = tf.reshape(bin_tensor, self.shape)
        return bin_tensor

class ImageClass():
    def __init__(self, shape, channels, ratio = None, name = None):
        self.channels = channels
        self.ratio = ratio
        self.name = name
        self.shape = shape
        self.decode_fun = None

    def decode(self, filename):
        image_tensor = tf.read_file(filename)
        image_tensor = self.decode_fun(image_tensor, channels = self.channels, ratio = self.ratio)
        image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
        image_tensor = tf.image.resize_images(image_tensor, 
                                        [self.shape[0] , self.shape[1]])
        
        return image_tensor


class JPGClass(ImageClass):
    def __init__(self, shape, channels = None, ratio = None, name = None):
        ImageClass.__init__(self, shape, channels, ratio, name)
        """ 
            used for load jpg image file
        """
        self.decode_fun = tf.image.decode_jpeg
		
class PNGClass(ImageClass):
    """ 
        used for load png image file
    """
    def __init__(self, shape, channels = None, ratio = None, name = None):
        ImageClass.__init__(self, shape, channels, ratio, name)
        self.decode_fun = tf.image.decode_png

