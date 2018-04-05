# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 09:56:11 2018

@author: sorrav
"""

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# from tensorflow.examples.tutorials.mnist import input_data
from dataset_NEWtf import load_dataset
# Code for saving and restoring the model
import SaveRestoreReset as srr

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

FLAGS = None

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
  maxSources = data_params[2]
  numFrames = data_params[1]
  imgSize = data_params[0]
  with tf.name_scope('reshape_x'):
    if debug:
      print("reshape_x:")
    x_image = tf.reshape(x, [-1, imgSize, imgSize, numFrames])
    # -1 is for the batch size, will be dynamically assigned 

  # First convolutional layer - maps numFrames to 16 features.
  with tf.name_scope('conv1'):
    if debug:
      print("conv1:")
    W_conv1 = weight_variable([5, 5, numFrames, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X to 32x32.
  with tf.name_scope('pool1'):
    if debug:
      print("pool1:")
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 16 feature maps to 32.
  with tf.name_scope('conv2'):
    if debug:
      print("conv2:")
    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer to 16x16.
  with tf.name_scope('pool2'):
    if debug:
      print("pool2:")
    h_pool2 = max_pool_2x2(h_conv2)
    
  # Third convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv3'):
    if debug:
      print("conv3:")
    W_conv3 = weight_variable([5, 5, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  # Third pooling layer to 8x8.
  with tf.name_scope('pool3'):
    if debug:
      print("pool3:")
    h_pool3 = max_pool_2x2(h_conv3)

#  # Fully connected layer 1 -- after 2 round of downsampling, our 100x100 image
#  # is down to 25x25x64 feature maps -- maps this to 65536 features.
#  fc1_length = 8192
#  with tf.name_scope('fc1'):
#    if debug:
#      print("fc1:")
#    W_fc1 = weight_variable([int(imgSize/4) * int(imgSize/4) * 16, fc1_length])
#    b_fc1 = bias_variable([fc1_length])
#
#    h_pool2_flat = tf.reshape(h_pool2, [-1, int(imgSize/4)*int(imgSize/4)*16])
#    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#  # Dropout - controls the complexity of the model, prevents co-adaptation of
#  # features.
#  with tf.name_scope('dropout'):
#    if debug:
#      print("dropout:")
#    keep_prob = tf.placeholder(tf.float32)
#    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#  # Map the 1024 features to 10 classes, one for each digit
#  with tf.name_scope('fc2'):
#    if debug:
#      print("fc2:")
#    W_fc2 = weight_variable([fc1_length, imgSize*imgSize*maxSources])
#    b_fc2 = bias_variable([imgSize*imgSize*maxSources])
#
#    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  # First deconvolutional layer - maps 64 to 32 features.
  with tf.name_scope('deconv1'):
    if debug:
      print("deconv1:")
    W_deconv1 = weight_variable([5, 5, 32, 64])
    b_deconv1 = bias_variable([32])
    h_deconv1 = tf.nn.relu(conv2d_transpose(x_image, W_deconv1) + b_deconv1) # need to add encoder output instead of x_image
   
  # Unpooling layer - upsamples by 2X to 16x16. 
  with tf.name_scope('unpool1'): 
    if debug:
      print("unpool1:")
    h_unpool1 = unpool_2d(h_deconv1)
    
  # Second deconvolutional layer - maps 32 to 16 features.
  with tf.name_scope('deconv2'):
    if debug:
      print("deconv2:")
    W_deconv2 = weight_variable([5, 5, 16, 32])
    b_deconv2 = bias_variable([16])
    h_deconv2 = tf.nn.relu(conv2d_transpose(h_unpool1, W_deconv2) + b_deconv2) 
   
  # Unpooling layer - upsamples by 2X to 32x32. 
  with tf.name_scope('unpool2'): 
    if debug:
      print("unpool2:")
    h_unpool2 = unpool_2d(h_deconv2)
    
  # 3.1 deconvolutional layer - maps 16 to 8 features.
  with tf.name_scope('deconv31'):
    if debug:
      print("deconv31:")
    W_deconv31 = weight_variable([5, 5, 8, 16])
    b_deconv31 = bias_variable([8])
    h_deconv31 = tf.nn.relu(conv2d_transpose(h_unpool2, W_deconv31) + b_deconv31)

  # 3.2 deconvolutional layer - maps 8 to 2 features.
  with tf.name_scope('deconv32'):
    if debug:
      print("deconv32:")
    W_deconv32 = weight_variable([5, 5, 2, 8])
    b_deconv32 = bias_variable([2])
    h_deconv32 = tf.nn.relu(conv2d_transpose(h_deconv31, W_deconv32) + b_deconv32)
   
  # Unpooling layer - upsamples by 2X to 64x64. 
  with tf.name_scope('unpool3'): 
    if debug:
      print("unpool3:")
    y_conv = unpool_2d(h_deconv32)
    
  with tf.name_scope('reshape_y'):
    if debug:
      print("reshape_y:")
    y_conv = tf.reshape(y_conv, [-1, imgSize, imgSize, maxSources])
    # -1 is for the batch size, will be dynamically assigned 
    
  return y_conv, keep_prob

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  # filter: A 4-D Tensor with the same type as value and shape [height, width, in_channels, out_channels]
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_transpose(x, W):
  """conv2d_transpose returns a 2d deconvolution layer with full stride."""
  # filter: A 4-D Tensor with the same type as value and shape [height, width, out_channels, in_channels]  
  return tf.nn.conv2d_transpose(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def unpool_2d(pool, 
              ind, 
              stride=[1, 2, 2, 1], 
              scope='unpool_2d'):
  """Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366
  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
  """    
  with tf.variable_scope(scope):
    input_shape = tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                      shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  if debug:
    print("weight_variable:")
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  if debug:
    print("bias_variable:")
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def cross_corr(logits, labels, batch_size, data_params):
    maxSources = data_params[2]
    imgSize = data_params[0]
    if debug:
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
  try:
      # Save Graph and Checkpoints
      srr.reset()
      file_path = os.path.dirname(os.path.abspath(__file__))
      graph_location = os.path.join(file_path,'graphs','graph_im64_f8_s4')
      ckpt_location = os.path.join(file_path,'checkpoints','ckpt_im64_f8_s4')
      model_name = 'im64_f8_s4'
      restored_ckpt_name = 'im64_f8_s4_2018-04-04_1615' # for name mode in restore
      if not os.path.exists(ckpt_location):
        os.makedirs(ckpt_location)
      # Restore params  
      restoreFlag = 1  
      mode = 'last' #last - take last checkpoint, name - get apecific checkpoint by name, best - take checkpoint with best accuracy so far (not supported yet)
      
      # Manage checkpoints log
      log_obj = srr.get_log(ckpt_location, model_name)
      log_obj.write('\n' + ('#' * 50))
      ckpt_start_time = srr.get_time()
      log_obj.write("\ncheckpoint name: %s" % model_name + '_' + ckpt_start_time)
      
      with tf.name_scope('data'):  
          # Import data
          first_sample = 1
          num_samp = 25
          dataObj, imgSize, numFrames, maxSources = load_dataset(first_sample,num_samp)
          data_params = [imgSize, numFrames, maxSources]
          print("loaded data with the following params:")
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
        loss = tf.reduce_mean(tf.losses.mean_squared_error(y_,y_conv))
    
      with tf.name_scope('adam_optimizer'):
        lr = 1e-4
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    
      with tf.name_scope('accuracy'):
         accuracy = cross_corr(y_conv, y_, batch_size, data_params)
         
      # Create Summaries
      tf.summary.scalar("loss", loss)
      tf.summary.scalar("accuracy", accuracy)            
      # because you have several summaries, we should merge them all
      # into one op to make it easier to manage
      summary_op = tf.summary.merge_all()
    
      with tf.Session() as sess:
          
        sess.run(tf.global_variables_initializer())

        # Create writer objects
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location, graph=tf.get_default_graph())            
        
        if restoreFlag:
            res_name = srr.restore(sess, ckpt_location, restored_ckpt_name, mode)
            log_obj.write("\nrestored from: %s" % res_name)
            
        for i in range(num_samp):
          batch = dataObj.train.next_batch(batch_size)
          _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
          if i % 5 == 0: 
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            train_writer.add_summary(summary, i)          
         
        log_obj.write("\ntrain accuracy: %s" % accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
        log_obj.write("\nfinished: %s" % srr.get_time())
        log_obj.close()
        # Saving checkpoints
        srr.save(sess, ckpt_location, model_name + '_' + ckpt_start_time)
        # Saving graph
        train_writer.close() 
          
    ###########     start test section:
        for_print = sess.run(y_conv, feed_dict={x: batch[0], keep_prob: 0.5})
        for_print= for_print[0, :, :, 1]
        plt.figure(1)
        plt.imshow(for_print)
        y_img = batch[1][0, :, :, 1]
        plt.figure(2)
        plt.imshow(y_img)
    ###########      end test section:
    
        print('test accuracy %g' % accuracy.eval(feed_dict={
                x: dataObj.test.features, y_: dataObj.test.labels, keep_prob: 1.0}))
                
  except Exception:
      log_obj.close()
      train_writer.close() 

if __name__ == '__main__':
  debug= False
  if debug:
    print("started")
  tf.app.run(main=main, argv=[sys.argv[0]])
  