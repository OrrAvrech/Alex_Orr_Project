# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:38:46 2017

@author: AOP
"""
#%% Model specs

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py

ImgSize = 200
NumPix = ImgSize * ImgSize 

x = tf.placeholder(tf.float32, [None*NumPix, None])
W = tf.Variable(tf.zeros([None, None]))
b = tf.Variable(tf.zeros([]))
y = tf.nn.softmax(tf.matmul(x, W, False, True) + b)

# Data tags
y_ = tf.placeholder(tf.float32, [None, Num_classes])

#%% Model Training and Evaluation by Test set

# Risk - should be minimized
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Stochastic training - using batches of 100 training samples
for _ in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #max idx along 10 columns
# Num of correct predictions div. by num of samples
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#%% Choose parameters using mnist.validation

# Optimal Gradient Step
#grad_step = np.array([0.01, 0.1, 0.5, 1])
#acc_vec = np.zeros(np.size(grad_step), dtype = np.float32)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#for i in range(0,np.size(grad_step)):
#    train_step = tf.train.GradientDescentOptimizer(grad_step[i]).minimize(cross_entropy)
#    sess = tf.InteractiveSession()
#    tf.global_variables_initializer().run()
#    for _ in range(2000):
#        batch_xs, batch_ys = mnist.validation.next_batch(100)
#        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#        
#    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
#    sess_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#    acc_vec[i] = sess_acc
##    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#    
#plt.plot(grad_step, acc_vec)
#plt.ylabel('Accuracy')
#plt.show()    
