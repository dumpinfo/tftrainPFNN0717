import numpy as np
import tensorflow as tf
import PFNNParameter as PFNN
import Utils as utils
from PFNNParameter import PFNNParameter
from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer
import os.path

tf.set_random_seed(23456)  

utils.build_path(['data'])
utils.build_path(['training'])
utils.build_path(['training/nn'])
utils.build_path(['training/weights'])
utils.build_path(['training/model'])
'''
X = np.float32(np.loadtxt('./data/Input.txt'))
Y = np.float32(np.loadtxt('./data/Output.txt'))

P = np.expand_dims(X[...,-1], axis =1)
X = np.delete(X, -1, axis=1)

Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

for i in range(Xstd.size):
    if (Xstd[i]==0):
        Xstd[i]=1
for i in range(Ystd.size):
    if (Ystd[i]==0):
        Ystd[i]=1
        
X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd

Xmean.tofile('training/nn/Xmean.bin')
Ymean.tofile('training/nn/Ymean.bin')
Xstd.tofile('training/nn/Xstd.bin')
Ystd.tofile('training/nn/Ystd.bin')
'''
X = np.load('X.npy')
Y = np.load('Y.npy')
P = np.load('P.npy')
P = np.reshape(P,(X.shape[0],1))#X

X = np.concatenate((X, P), axis=1)

Xdim = X.shape[1]
Ydim = Y.shape[1]
samples = X.shape[0]

rng = np.random.RandomState(23456)
###""" Phase Function Neural Network """
###"""input of nn"""

keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing 
num_gpus = 2
learning_rate      = 0.0001
weightDecay        = 0.0025

batch_size         = 32
training_epochs    = 150
Te                 = 10
Tmult              = 2
total_batch        = int(samples / batch_size)

I = np.arange(samples)
rng.shuffle(I)

#training set and  test set
count_test     = 0
num_testBatch  = np.int(total_batch * count_test)
num_trainBatch = total_batch - num_testBatch
print("training_batch:", num_trainBatch)
print("test_batch:", num_testBatch)


def PFNNet(X_nn, keep_prob , reuse ):
  with tf.variable_scope('ConvNet', reuse=reuse):
     ####parameter of nn                                                                                           
     #rng = np.random.RandomState(23456)                                                                                 
     nslices = 4                             # number of control points in phase function                               
     phase = X_nn[:,-1]                      #phase                                                                     
     P0 = PFNNParameter((nslices, 512, Xdim-1), rng, phase, 'wb0')                                                      
     P1 = PFNNParameter((nslices, 512, 512), rng, phase, 'wb1')                                                         
     P2 = PFNNParameter((nslices, Ydim, 512), rng, phase, 'wb2')                                                        
                                                                                                                                            
     H0 = X_nn[:,:-1] 
     H0 = tf.expand_dims(H0, -1)       
     H0 = tf.nn.dropout(H0, keep_prob=keep_prob )#, training=is_training)
     
     b0 = tf.expand_dims(P0.bias, -1)      
     H1 = tf.matmul(P0.weight, H0) + b0      
     H1 = tf.nn.elu(H1)             
     H1 = tf.nn.dropout(H1, keep_prob=keep_prob )#, training=is_training) 
     
     b1 = tf.expand_dims(P1.bias, -1)       
     H2 = tf.matmul(P1.weight, H1) + b1       
     H2 = tf.nn.elu(H2)                
     H2 = tf.nn.dropout(H2, keep_prob=keep_prob )#, training=is_training) 
     
     b2 = tf.expand_dims(P2.bias, -1)       
     H3 = tf.matmul(P2.weight, H2) + b2      
     H3 = tf.squeeze(H3, -1)
  return H3          

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
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
    return average_grads

    
# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

####=======================================================================================



###########################################################################################
# Place all ops on CPU by default
with tf.device('/cpu:0'):
    tower_grads = []
    reuse_vars = False

    # tf Graph input
    #X = tf.placeholder(tf.float32, [None, num_input])
    #Y = tf.placeholder(tf.float32, [None, num_classes])
    X_nn = tf.placeholder(tf.float32, [None, Xdim], name='x-input')
    Y_nn = tf.placeholder(tf.float32, [None, Ydim], name='y-input')

    # Loop over all GPUs and construct their own computation graph
    for i in range(num_gpus):
        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):

            # Split data between GPUs
            _x = X[i * batch_size: (i+1) * batch_size]
            _y = Y[i * batch_size: (i+1) * batch_size]

            # Because Dropout have different behavior at training and prediction time, we
            # need to create 2 distinct computation graphs that share the same weights.

            # Create a graph for training
            #logits_train = conv_net(_x, num_classes, dropout,
            #                        reuse=reuse_vars, is_training=True)
            keep_prob = 0.8
            pfnny_train =   PFNNet(_x, keep_prob , reuse=reuse_vars)#, is_training=True)                      
            keep_prob = 1.0
            # Create another graph for testing that reuse the same weights
            pfnny_test  =   PFNNet(_x, keep_prob , reuse=reuse_vars)#, is_training=False) 

            # Define loss and optimizer (with train logits, for dropout to take effect)
            #loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #    logits=logits_train, labels=_y))
            loss_op = tf.reduce_mean(tf.square(_y - pfnny_train))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss_op)

            # Only first GPU compute accuracy
            if i == 0:
                # Evaluate model (with test logits, for dropout to be disabled)
                #correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                accuracy = tf.reduce_mean(tf.square(_y - pfnny_test))

            reuse_vars = True
            tower_grads.append(grads)
    print("hello test!",len(tower_grads))
    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start Training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Keep training until reach max iterations
        for i in range(num_trainBatch):
            index_train = I[i*batch_size*num_gpus:(i+1)*batch_size*num_gpus]
            batch_xs = X[index_train]
            batch_ys = Y[index_train]
            ##=======================================================
            feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.8}
            testError = sess.run(loss, feed_dict=feed_dict)
            avg_cost_test += testError / num_testBatch
            if i % 1000 == 0:
                print(i, "testloss:",testError)
            ##=======================================================
            ts = time.time()
            sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys})
            te = time.time() - ts
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_xs,
                                                                     Y: batch_ys})
                print("Step " + str(step) + ": Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc) + ", %i Examples/sec" % int(len(batch_xs)/te))
            step += 1
            #save_path = saver.save(sess, "training/model/model.ckpt")
            #PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
            #           (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
            #            50, 
            #            'training/nn'
            #            )
        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        #print("Testing Accuracy:", \
        #    np.mean([sess.run(accuracy, feed_dict={X: mnist.test.images[i:i+batch_size],
        #    Y: mnist.test.labels[i:i+batch_size]}) for i in range(0, len(mnist.test.images), batch_size)]))


###########################################################################################


  
#-----------------------------above is model training----------------------------------












