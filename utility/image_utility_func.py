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

def resize_keep_ratio(image, target_width, target_height):
    """ Resize image and keep aspect ratio.
    Args:
        image:
        target_width:
        target_hegiht:
    """
    h, w, c = image.shape
    target_image = np.zeros((target_height, target_width, c), image.dtype)
    target_ratio = target_width / float(target_height)
    image_ratio = h / float(w)
    if target_ratio > image_ratio:
        ratio = target_width / float(w)
        new_w = target_width
        new_h = int(h * ratio)
        image = cv2.resize(image, (new_w, new_h)) 
        h_offset = int((target_height - new_h)/2.0)
        target_image[h_offset: new_h + h_offset, :, :] = image
    else:
        ratio = target_height / float(h)
        new_w = int(w * ratio)
        new_h = target_height
        image = cv2.resize(image, (new_w, new_h)) 
        w_offset = int((target_width - new_w)/2.0)
        target_image[:, w_offset: new_w + w_offset, :] = image

    return target_image

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
            arg_list_copy[i] = to_color(arg_list_copy[i]) 

    return tf.concat(arg_list_copy, dim, name = "concat_image")

def merge_image_np(dim, arg_list):
    arg_list_copy = [arg for arg in arg_list]
    def to_color(image_gray):
        image_color = np.tile(image_gray, [1,1,1,3])
        return image_color
    def np_norm_image(image):
        image = (image - np.amin(image)) / \
                    (np.amax(image) - np.amin(image))
        return image

    for i, arg in enumerate(arg_list_copy):
        arg_list_copy[i] = np_norm_image(arg)
        if arg.shape[3] == 1:
            arg_list_copy[i] = to_color(arg_list_copy[i]) 

    return np.concatenate(arg_list_copy, dim)

def batch_center_crop_frac(batch_image, frac):
    """
    Args:
        batch_image: [b, h, w, c]
        frac: 0.5, 0.75

    """
    b, h, w, c =  batch_image.get_shape().as_list()
    start_h = int((h - frac * h)/2)
    start_w = int((w - frac * w)/2)
    
    end_h = start_h + int(frac * h)
    end_w = start_w + int(frac * w)

    croped_image = batch_image[:, start_h:end_h, start_w:end_w,:]

    return croped_image

