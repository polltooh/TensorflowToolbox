import tensorflow as tf


def average_gradients(grads_list, loss_list):
    """Calculate the average gradient for each shared variable across all towers.
    
    Note that this function provides a synchronization point across all towers.
    
    Args:
    grads_list: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    with tf.variable_scope('average_grads'):
        average_loss = tf.reduce_mean(loss_list)
        average_grads = []
        for grad_and_vars in zip(*grads_list):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
        
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

    return average_loss, average_grads

def var_getter(device):
    def custom_getter(getter, name, *args, **kwargs):
        with tf.device(device):
            return getter(name, *args, **kwargs)

def single_grad(model, opt, batch_data, is_train, scope, reuse):
    custom_getter = var_getter('/cpu:0')
    with tf.name_scope(scope), tf.variable_scope('', custom_getter=custom_getter, reuse=reuse):
        with tf.variable_scope('network'):
            output = model.model_infer(batch_data, is_train)
        model_loss = model.model_loss(batch_data, output)
        tf.add_to_collection("losses", model_loss)

        # The losses will also store weight decay loss.
        loss = tf.add_n(tf.get_collection("losses", scope), name="total_loss")

        if is_train:
            grads = opt.compute_gradients(loss)
        else:
            grads = None
    return loss, grads 

def multi_grads(model, num_gpus, train_input=None, test_input=None):
    grads_list = list()
    loss_list = list()
    opt = model.model_optimizer()
    scope = model.scope

    grads = None
    loss = None
    test_loss = None

    is_train = False
    if train_input is not None: 
        is_train = True
        if num_gpus >= 1:
            for i in xrange(num_gpus):
                with tf.device('/gpu:%d' % i):
                    loss, grads = single_grad(model, opt, train_input[i], is_train,
                                        "%s_train_%i"%(scope,i), reuse=(i>0))
                    loss_list.append(loss)
                    grads_list.append(grads)

        else:
            with tf.device('/cpu'):
                loss, grads = single_grad(model, opt, train_input, is_train, 
                                          "%s_train"%(scope), reuse=False)
                grads_list.append(grads)
                loss_list.append(loss)

        loss, grads = average_gradients(grads_list, loss_list)

    if test_input is not None:
        reuse = is_train
        if num_gpus >= 1:
            test_loss_list = list()
            for i in xrange(num_gpus):
                with tf.device('/gpu:%d' % 0):
                    test_loss, test_grads = single_grad(model, opt, test_input[i], False, 
                                                        "%s_test_%i"%(scope, i), reuse)
                    test_loss_list.append(test_loss)
            test_loss = tf.reduce_mean(test_loss_list)
        else:
            with tf.device('/cpu:0'):
                test_loss, _ = single_grad(model, opt, test_input, False, 
                                                    "%s_test"%(scope), reuse)

    return loss, grads, test_loss

def model_trainer(model, num_gpus, train_input=None, test_input=None):
    """ Function for training the model.


    """
    with tf.device('/cpu:0'):
        train_op = None
        loss = None
        test_loss = None

        loss, grads, test_loss = multi_grads(model, num_gpus, train_input, test_input)
        if grads is not None:
            opt = model.model_optimizer()
            with tf.variable_scope('trainer'):
                global_step = tf.get_variable('global_step', [], 
                                              initializer=tf.constant_initializer(0),
                                              trainable=False)

                apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

                variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())

                train_op = tf.group(apply_gradient_op, variables_averages_op)

    return train_op, loss, test_loss
