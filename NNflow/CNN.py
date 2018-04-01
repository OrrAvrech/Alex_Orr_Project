# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 09:56:11 2018

@author: sorrav
"""

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

#from tensorflow.examples.tutorials.mnist import input_data
from dataset_NEWtf import load_dataset

import tensorflow as tf
import os
import numpy as np
from scipy import signal

FLAGS = None

# Constant Dimensions -- Global
#imgSize = 12
#numFrames = 3  
#maxSources = 15

def deepnn(x,data_params):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - it would be 1 for grayscale
  # 3 for an RGB image, 4 for RGBA, numFrames for a movie.    
  maxSources = data_params["maxSources"]
  imgSize = data_params["imgSize"]
  numFrames = data_params["numFrames"]
  with tf.name_scope('reshape_x'):
    if __debug__:
      print("reshape_x:")
    x_image = tf.reshape(x, [-1, imgSize, imgSize, numFrames])
    # -1 is for the batch size, will be dynamically assigned 

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    if __debug__:
      print("conv1:")
    W_conv1 = weight_variable([5, 5, numFrames, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    if __debug__:
      print("pool1:")
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    if __debug__:
      print("conv2:")
    W_conv2 = weight_variable([5, 5, 16, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    if __debug__:
      print("pool2:")
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  # Fully connected layer 1 -- after 2 round of downsampling, our 100x100 image
  # is down to 25x25x64 feature maps -- maps this to 65536 features.
  fc1_length = 2048
  with tf.name_scope('fc1'):
    if __debug__:
      print("fc1:")
    W_fc1 = weight_variable([int(imgSize/4) * int(imgSize/4) * 16, fc1_length])
    b_fc1 = bias_variable([fc1_length])

    h_pool2_flat = tf.reshape(h_pool2, [-1, int(imgSize/4)*int(imgSize/4)*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    if __debug__:
      print("dropout:")
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    if __debug__:
      print("fc2:")
    W_fc2 = weight_variable([fc1_length, imgSize*imgSize*maxSources])
    b_fc2 = bias_variable([imgSize*imgSize*maxSources])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
  with tf.name_scope('reshape_y'):
    if __debug__:
      print("reshape_y:")
    y_conv = tf.reshape(y_conv, [-1, imgSize, imgSize, maxSources])
    # -1 is for the batch size, will be dynamically assigned 
    
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  if __debug__:
    print("conv2d:")
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  if __debug__:
    print("max_pool_2x2:")
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  if __debug__:
    print("weight_variable:")
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  if __debug__:
    print("bias_variable:")
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def cross_corr(logits, labels, batch_size, data_params):
    maxSources = data_params["maxSources"]
    imgSize = data_params["imgSize"]
    if __debug__:
      print("cross_corr:")
    for i in range(batch_size):  
        y_conv = logits[i, 0:imgSize, 0:imgSize, 0:maxSources]
        y_conv = tf.reshape(y_conv, [1, imgSize, imgSize, maxSources])
        label_resh = tf.reshape(labels, [imgSize, imgSize, maxSources, -1])
        y_ = label_resh[0:imgSize, 0:imgSize, 0:maxSources, i]
        y_ = tf.reshape(y_, [imgSize, imgSize, maxSources, 1])
        result = 0  
        corr2d = tf.nn.conv2d(y_conv, y_, strides=[1, 1, 1, 1], padding='SAME')
        result += tf.reduce_mean(corr2d)
    return result/batch_size

def main(_):
     
  # Append the number of training steps our model has gone through
  # In a variable called global_step
  global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
    
  with tf.name_scope('data'):  
      # Import data
      dataObj, imgSize, numFrames, maxSources = load_dataset()
      data_params = {"imgSize":imgSize, "numFrames":numFrames, "maxSources":maxSources}
      print("loaded data")
      print("imgSize is:" +str(imgSize))
      print("numFrames is:" +str(numFrames))
      print("maxSources is:" +str(maxSources))
      batch_size = 1

  # Create the model
  x = tf.placeholder(tf.float32, [None, imgSize, imgSize, numFrames])
  y_ = tf.placeholder(tf.float32, [None, imgSize, imgSize, maxSources])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x,data_params)

  # Define loss and optimizer
  with tf.name_scope('loss'):
#    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
#                                                            logits=y_conv)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(y_,y_conv))
#    cost = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=1))
#  cross_entropy = tf.reduce_mean(cross_entropy)
    

  with tf.name_scope('adam_optimizer'):
    lr = 1e-4
    train_step = tf.train.AdamOptimizer(lr).minimize(cost, global_step=global_step)
#  with tf.name_scope('accuracy'):
#    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#    correct_prediction = tf.cast(correct_prediction, tf.float32)
#  accuracy = tf.reduce_mean(correct_prediction)
    
  with tf.name_scope('accuracy'):
     accuracy = cross_corr(y_conv, y_, batch_size, data_params)

  # Save Graph and Checkpoints
  file_path = os.path.dirname(os.path.abspath(__file__))
  graph_location = file_path + '\\graphs\\graph_im64_f8_s2\\'
  ckpt_location = file_path + '\\checkpoints\\ckpt_im64_f8_s2\\'
  if not os.path.exists(ckpt_location):
    os.makedirs(ckpt_location)
  saver = tf.train.Saver()
  #  train_writer.add_graph(tf.get_default_graph())

  # Obtain Summaries
  tf.summary.scalar("loss", cost)
  summary_op = tf.summary.merge_all() # Needed for many summaries  

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_location + 'checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)    
    
    for i in range(30):
      batch = dataObj.train.next_batch(batch_size)
      #train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      if i % 10 == 0: 
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        saver.save(sess, ckpt_location + 'im64_f8_s2', global_step=i)
        loss_batch, _, summary = sess.run([cost, train_step, summary_op])
        train_writer.add_summary(summary, global_step=i)
       
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: dataObj.test.features, y_: dataObj.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
#  parser = argparse.ArgumentParser()
#  parser.add_argument('--data_dir', type=str,
#                      default='/tmp/tensorflow/mnist/input_data',
#                      help='Directory for storing input data')
#  FLAGS, unparsed = parser.parse_known_args()
#  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  if __debug__:
    print("started")
  tf.app.run(main=main, argv=[sys.argv[0]])