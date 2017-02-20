import numpy as np
import cv2
import tensorflow as tf

CV_VERSION = cv2.__version__.split(".")[0]

def load_image(image_name):
    image = cv2.imread(image_name)
    return image

def norm_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image

def repeat_image(image):
    #image = np.expand_dims(image, 2)
    image = np.tile(image, (1,1,3))
    return image

def show_image(input_image, normalize = True, wait_time = 0, image_name = "image"):
    if normalize:
        input_image = norm_image(input_image)
    cv2.imshow(image_name, input_image)
    cv2.waitKey(wait_time)

def save_image(input_image, image_name, is_norm = False):
    if is_norm:
        input_image = norm_image(input_image)
    cv2.imwrite(image_name, input_image)
    #cv2.imwrite(FLAGS.image_dir + "/%08d.jpg"%(i/100), image)

def resize_image(image, rshape):
    """
    Args:
        image:
        rshape: (new_image_hegiht, new_image_width)
    """
    return cv2.resize(image, rshape)

def get_bbox(image, threshold_v):
    """
    Args:
        threshold_v: threshold value, normally 127
    Return:
        bbox: list [x, y, w, h], bounding box
    """

    ret,thresh = cv2.threshold(image, threshold_v, 255, 0)
    if CV_VERSION == '2':
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
    elif CV_VERSION == '3':
        im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    
    cnt = contours[0]
    bbox = cv2.boundingRect(cnt)
    return bbox 

def merge_image(dim, arg_list):
    """
    Args:
        dim: dimention need to be concated
        arg_list: list of 4D tensor
    Return:
        merged tensor
    """
    arg_list_copy = [tf.identity(arg) for arg in arg_list]
    def to_color(image_gray):
        image_color = tf.tile(image_gray, [1,1,1,3])
        return image_color

    def tf_norm_image(image):
        image = (image - tf.reduce_min(image)) / \
                    (tf.reduce_max(image) - tf.reduce_min(image))
        return image

    for i, arg in enumerate(arg_list_copy):
        arg_list_copy[i] = tf_norm_image(arg)
        if arg.get_shape().as_list()[3] == 1:
            arg_list_copy[i] = to_color(arg) 

    return tf.concat(dim, arg_list_copy, name = "concat_image")


