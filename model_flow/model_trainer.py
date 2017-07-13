import tensorflow as tf

def average_gradients(grads_list):
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
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
    
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def single_grad(model, opt):
    output = model.model_infer() 
    loss = model.model_loss(output)
    grads = opt.compute_gradients(loss)
    return grads 

def multi_grads(model, num_gpus):
    grads_list = list()
    opt = model.optimizer()
    
    if num_gpus >= 1:
        for i in xrange(num_gpus):
          with tf.device('/gpu:%d' % i):
            with tf.variable_scope('trainer_%d' % (i)):
                grads = single_grad(model, opt)
                grads_list.append(grads)

        grads = average_gradients(grads_list)
    else:
        with tf.device('/cpu'):
            grads = single_grad(model, opt)

    return grads

def model_trainer(model, num_gpus):
    grads = multi_grads(model, num_gpus)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)

    return train_op
