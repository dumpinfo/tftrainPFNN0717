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


def PFNNet(X_nn, keep_prob):
  #with tf.variable_scope('ConvNet', reuse=reuse):
     ####parameter of nn                                                                                           
                                                                              
     nslices = 4                             # number of control points in phase function                               
     phase = X_nn[:,-1]                      #phase                                                                     
      
     P0 = PFNNParameter((nslices, 512, Xdim-1), rng, phase, 'wb0')
     P1 = PFNNParameter((nslices, 512, 512), rng, phase, 'wb1')
     P2 = PFNNParameter((nslices, Ydim, 512), rng, phase, 'wb2')                                                   
                                                                                                                                            
     H0 = X_nn[:,:-1] 
     H0 = tf.expand_dims(H0, -1)       
     H0 = tf.nn.dropout(H0, keep_prob=keep_prob )
     
     b0 = tf.expand_dims(P0.bias, -1)      
     H1 = tf.matmul(P0.weight, H0) + b0      
     H1 = tf.nn.elu(H1)             
     H1 = tf.nn.dropout(H1, keep_prob=keep_prob ) 
     
     b1 = tf.expand_dims(P1.bias, -1)       
     H2 = tf.matmul(P1.weight, H1) + b1       
     H2 = tf.nn.elu(H2)                
     H2 = tf.nn.dropout(H2, keep_prob=keep_prob ) 
     
     b2 = tf.expand_dims(P2.bias, -1)       
     H3 = tf.matmul(P2.weight, H2) + b2      
     H3 = tf.squeeze(H3, -1)
     return H3 


X = np.load('XXa.npy')
Y = np.load('YYa.npy')
P = np.load('PPa.npy')
P = np.reshape(P,(X.shape[0],1))#X
X = np.concatenate((X, P), axis=1)
Xdim  = X.shape[1]
Ydim = Y.shape[1]
samples = X.shape[0]

learning_rate      = 0.0001
weightDecay        = 0.0025
batch_size         = 32
training_epochs    = 150
Te                 = 10
Tmult              = 2
total_batch        = int(samples / batch_size)




""" Phase Function Neural Network """
"""input of nn"""
########################################################################
X_nn = tf.placeholder(tf.float32, [None, Xdim], name='x-input')
Y_nn = tf.placeholder(tf.float32, [None, Ydim], name='y-input')

rng = np.random.RandomState(23456)   
keep_prob = tf.placeholder(tf.float32)
H3 = PFNNet(X_nn, keep_prob)
loss = tf.reduce_mean(tf.square(Y_nn - H3)) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



#session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#saver for saving the variables
saver = tf.train.Saver()


#batch size and epoch
print("total_batch:", total_batch)

#randomly select training set
I = np.arange(samples)
rng.shuffle(I)


#training set and  test set
count_test     = 0
num_testBatch  = np.int(total_batch * count_test)
num_trainBatch = total_batch - num_testBatch
print("training_batch:", num_trainBatch)
print("test_batch:", num_testBatch)

   
#used for saving errorof each epoch
error_train = np.ones(training_epochs)
error_test  = np.ones(training_epochs)

#start to train
print('Learning start..')
for epoch in range(training_epochs):
    avg_cost_train = 0
    avg_cost_test  = 0

    for i in range(num_trainBatch):
        index_train = I[i*batch_size:(i+1)*batch_size]
        batch_xs = X[index_train]
        batch_ys = Y[index_train]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7}
        l, _, = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost_train += l / num_trainBatch
        
        if i % 100 == 0:
            print(i, "trainingloss:", l)
            #print('Epoch:', '%04d' % (epoch + 1), 'clr:', clr)
            #print('Epoch:', '%04d' % (epoch + 1), 'wdc:', wdc)
            
    for i in range(num_testBatch):
        if i==0:
            index_test = I[-(i+1)*batch_size: ]
        else:
            index_test = I[-(i+1)*batch_size: -i*batch_size]
        batch_xs = X[index_test]
        batch_ys = Y[index_test]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 1}
        testError = sess.run(loss, feed_dict=feed_dict)
        avg_cost_test += testError / num_testBatch
        if i % 1000 == 0:
            print(i, "testloss:",testError)
    
    #print and save training test error 
    print('Epoch:', '%04d' % (epoch + 1), 'trainingloss =', '{:.9f}'.format(avg_cost_train))
    print('Epoch:', '%04d' % (epoch + 1), 'testloss =', '{:.9f}'.format(avg_cost_test))
    error_train[epoch] = avg_cost_train
    error_test[epoch]  = avg_cost_test
    error_train.tofile("training/model/error_train.bin")
    error_test.tofile("training/model/error_test.bin")
    
    #save model and weights
    save_path = saver.save(sess, "training/model/model.ckpt")
    PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
                      (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
                      50, 
                      'training/nn'
                      )    
    
    #save weights every 5epoch
    if epoch>0 and epoch%10==0:
        path_human  = 'training/weights/humanNN%03i' % epoch
        if not os.path.exists(path_human):
            os.makedirs(path_human)
        PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
                          (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
                          50, 
                          path_human)
      
print('Learning Finished!')
#-----------------------------above is model training----------------------------------












