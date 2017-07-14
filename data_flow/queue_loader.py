import tensorflow as tf

import data_queue

class QueueLoader(object):
    def __init__(self, input_layer, params, is_train, sess, coord):
        """ 
        a class input_layer should include:
            read_data:
            process_data:
            one data point at a time
        """

        self.input_layer = input_layer
        self.params = params
        self.is_train = is_train
        self.coord = coord
        self.thread_list = list()
        self.sess = sess
        self.batch_data = self._create_queue()

    def _init_queue(self, shuffle, name, queue_params):
        read_queue = data_queue.init_file_queue(
                                        shuffle, 
                                        queue_params['capacity'],
                                        queue_params['dtypes'],
                                        queue_params['shapes'],
                                        name,
                                        queue_params['min_after_dequeue'])
        return read_queue 

    def _create_queue(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('queue_runner'):
                shuffle = self.is_train
                rq_params = self.params.read_queue
                read_queue = self._init_queue(shuffle, 'read_queue', rq_params)
                read_queue_size = read_queue.size()

                tf.summary.scalar('read_queue_size', read_queue_size)

                read_queue_input = self.input_layer.read_data()

                enqueue_ops = read_queue.enqueue(read_queue_input)
                enqueue_list = [enqueue_ops] * rq_params['num_threads']
                self._run_queue(read_queue, enqueue_list, 'run_read_queue')

                pq_params = self.params.process_queue
                process_queue = self._init_queue(
                                                shuffle, 
                                                'process_queue', 
                                                pq_params)
                process_queue_size = process_queue.size()

                tf.summary.scalar('process_queue_size', process_queue_size)

                read_tensor = read_queue.dequeue()
                process_tensor = self.input_layer.process_data(read_tensor)

                enqueue_ops = [process_queue.enqueue(process_tensor)] 
                enqueue_list = [enqueue_ops] * pq_params['num_threads']
                self._run_queue(process_queue, enqueue_list, 'process_queue')

                batch_data = process_queue.dequeue_many(self.params.batch_size)

        return batch_data

    def _run_queue(self, queue, enqueue_list, name):
        with tf.name_scope(name, 'runner'):
            runner = tf.train.queue_runner.QueueRunner(queue, enqueue_list)
            self.thread_list.append(runner.create_threads(
                                self.sess, self.coord, daemon=True, start=True))
