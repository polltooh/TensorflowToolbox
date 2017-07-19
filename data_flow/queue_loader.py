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
        queue = data_queue.init_file_queue(
                                        shuffle, 
                                        queue_params['capacity'],
                                        queue_params['dtypes'],
                                        queue_params['shapes'],
                                        name,
                                        queue_params['min_after_dequeue'])
        return queue 

    def _create_queue(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('queue_runner'):
                shuffle = self.is_train
                rq_params = self.params.load_queue
                load_queue = self._init_queue(shuffle, 'load_queue', rq_params)
                load_queue_size = load_queue.size()

                tf.summary.scalar('load_queue_size', load_queue_size)

                load_queue_input = self.input_layer.read_data(rq_params['dtypes'])

                enqueue_ops = load_queue.enqueue(load_queue_input)
                enqueue_list = [enqueue_ops] * rq_params['num_threads']
                self._run_queue(load_queue, enqueue_list, 'run_load_queue')

                pq_params = self.params.preprocess_queue
                preprocess_queue = self._init_queue(
                                                shuffle, 
                                                'preprocess_queue', 
                                                pq_params)
                preprocess_queue_size = preprocess_queue.size()

                tf.summary.scalar('preprocess_queue_size', preprocess_queue_size)

                read_tensor = load_queue.dequeue()
                process_tensor = self.input_layer.process_data(read_tensor, pq_params['dtypes'])

                enqueue_ops = preprocess_queue.enqueue(process_tensor) 
                enqueue_list = [enqueue_ops] * pq_params['num_threads']
                self._run_queue(preprocess_queue, enqueue_list, 'preprocess_queue')

                batch_data = preprocess_queue.dequeue_many(self.params.batch_size)

        return batch_data

    def _run_queue(self, queue, enqueue_list, name):
        with tf.name_scope(name, 'runner'):
            runner = tf.train.queue_runner.QueueRunner(queue, enqueue_list)
            self.thread_list.append(runner.create_threads(
                                self.sess, self.coord, daemon=True, start=True))
