def train(FLAGS, dataqueue, loss):
    #stdata = st_data.STData(file_name)
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    feature_ph = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.feature_dim))
    label_ph = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.label_dim))
    keep_prob_ph = tf.placeholder(tf.float32)
    train_test_phase_ph = tf.placeholder(tf.bool, name = 'phase_holder')

    infer = nt.inference(feature_ph, FLAGS.label_dim, keep_prob_ph, train_test_phase_ph)
    loss = nt.loss(infer, label_ph)
    train_op = nt.train_op(loss, FLAGS.init_learning_rate, global_step)

    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    for i in xrange(FLAGS.max_training_iter):
        feet_data = dataqueue.get_next_dict(train = True)
        #train_data, train_label = dataqueue.get_next_batch(FLAGS.batch_size)
        feed_data = {feature_ph:train_data, 
                label_ph:train_label, 
                keep_prob_ph:1, 
                train_test_phase_ph:True}
        
        _, loss_v, infer_v = sess.run([train_op, loss, infer], feed_data)

        if i % 100 == 0:
            print("i: %d loss: %f"%(i, loss_v))


