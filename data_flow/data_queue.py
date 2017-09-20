import tensorflow as tf


def init_file_queue(shuffle, capacity, dtypes, shapes, name, min_after_dequeue):
    if shuffle:
        file_queue = tf.RandomShuffleQueue(
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                dtypes=dtypes,
                shapes=shapes,
                name=name)
    else:
        file_queue = tf.FIFOQueue(
                capacity=capacity,
                dtypes=dtypes,
                shapes=shapes,
                name=name)

    return file_queue
