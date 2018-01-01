# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:38:46 2017

@author: AOP
"""
#%% Model specs

from tensorflow.examples.tutorials.mnist import input_data
#from .DataSimulation\final_Dataset_1 import 1
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #to_do:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as spio
from scipy.io import loadmat


import os
import sys

data_dir = os.path.dirname("P:\\Alex_Orr_Project\\DataSimulation\\final_Dataset_1\\")
data_file = os.path.join(data_dir, "1.mat")

with h5py.File(data_file, 'r') as f:
#    emitters_x = np.array(f.get('emitters/x'))
#    emitters_y = np.array(f.get('emitters/y'))
#    emitters_z = np.array(f.get('emitters/ZposIndex'))
#    zVec = np.array(f.get('emitters/zVec'))
#    emitters_x = [item for sublist in emitters_x for item in sublist] ##flatten list
#    emitters_y = [item for sublist in emitters_y for item in sublist] ##flatten list
#    emitters_z = [item for sublist in emitters_z for item in sublist] ##flatten list
#    zVec = [item for sublist in zVec for item in sublist] ##flatten list
#    print ('emitters_x is:\n' + str(emitters_x))
#    print ('emitters_y is:\n' + str(emitters_y))    
#    print ('emitters_z is:\n' + str(emitters_z))
#    print ('zVec is:\n' + str(zVec))    
    sample_x = np.array(f.get('x'))
    
    
    
    sample_y = np.array(f.get('y'))
    sample_y = np.reshape(sample_y, (1,sample_y.shape[0],sample_y.shape[1],sample_y.shape[2]))#
    sample_x = np.reshape(sample_x, (1,sample_x.shape[0],sample_x.shape[1],sample_x.shape[2]))#
    print ('sample_x shape is:' + str(sample_x.shape))
    print ('sample_y shape is:' + str(sample_y.shape))
#    sample_x = [item for sublist in sample_x for item in sublist] ##flatten list
#    sample_y = [item for sublist in sample_y for item in sublist] ##flatten list
#    print ('sample_x is:\n' + str(sample_x))
#    print ('sample_y is:\n' + str(sample_y))    

dataset_x = sample_x
dataset_y = sample_y

for i in range(2,11):
    data_file = os.path.join(data_dir, str(i)+ '.mat')
    print (data_file)
#    test = loadmat(data_file)
#    print (test)
    with h5py.File(data_file, 'r') as f:
        sample_x = np.array(f.get('x'))
        sample_y = np.array(f.get('y'))
        sample_y = np.reshape(sample_y, (1,sample_y.shape[0],sample_y.shape[1],sample_y.shape[2]))#
        sample_x = np.reshape(sample_x, (1,sample_x.shape[0],sample_x.shape[1],sample_x.shape[2]))#
#        sample_x = [item for sublist in sample_x for item in sublist] ##flatten list
#        sample_y = [item for sublist in sample_y for item in sublist] ##flatten list
#        print(sample_x)
#        print(sample_y)
        print(sample_x.shape)
        print(dataset_x.shape)
        dataset_x = np.append(dataset_x,sample_x, axis=0)
        dataset_y = np.append(dataset_y,sample_y, axis=0)
        print(dataset_x.shape)
        print(dataset_x)
#        dataset_x = np.c_[[dataset_x], [sample_x]]
features = dataset_x
labels = dataset_y
data = np
np.savez('/tmp/123.npz', features=features, labels=labels)
data = np.load('/tmp/123.npz')
tf.data.Dataset.from_tensor_slices((features, labels))


##    x = [item for sublist in x for item in sublist] ##flatten list
#    emitters =  np.array(f.get('emitters'))
#    print (emitters[2])
#    print(type(x[1]))
#    print(f.keys())
#    print(f['emitters'])
#    emitters_ = f['emitters']
#    print(emitters_.items())
##mat = spio.loadmat(data_file, squeeze_me=True)
#
## flat list 2 times faster:
### itertools.chain.from_iterable : $ python -mtimeit -s'from itertools import chain; l=[[1,2,3],[4,5,6], [7], [8,9]]*99' 'list(chain.from_iterable(l))'
#
#ImgSize = 100
#numFrames = 20
#maxSources = 5
#Num_classes = maxSources
#NumPix = ImgSize * ImgSize 
#
#x = tf.placeholder(tf.float32, [None, ImgSize, ImgSize, numFrames])
#W = tf.Variable(tf.zeros([None, None])) #to_do: choose dimensions
#b = tf.Variable(tf.zeros([])) #to_do: 
#y = tf.nn.softmax(tf.matmul(x, W, False, True) + b) 
#
## Data tags
#y_ = tf.placeholder(tf.float32, [None, ImgSize,ImgSize, Num_classes]) 
#
##%% Model Training and Evaluation by Test set
#
## Risk - should be minimized
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()
## Stochastic training - using batches of 100 training samples
#for _ in range(2000):
#  batch_xs, batch_ys = mnist.train.next_batch(100)
#  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
## Evaluation
##correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #max idx along 10 columns #to_do: insert success criterion
## Num of correct predictions div. by num of samples
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #to_do:
#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#
##%% Choose parameters using mnist.validation
#
## Optimal Gradient Step
##grad_step = np.array([0.01, 0.1, 0.5, 1])
##acc_vec = np.zeros(np.size(grad_step), dtype = np.float32)
##cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
##for i in range(0,np.size(grad_step)):
##    train_step = tf.train.GradientDescentOptimizer(grad_step[i]).minimize(cross_entropy)
##    sess = tf.InteractiveSession()
##    tf.global_variables_initializer().run()
##    for _ in range(2000):
##        batch_xs, batch_ys = mnist.validation.next_batch(100)
##        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
##        
##    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
##    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
##    sess_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
##    acc_vec[i] = sess_acc
###    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
##    
##plt.plot(grad_step, acc_vec)
##plt.ylabel('Accuracy')
##plt.show()    
