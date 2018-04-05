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
import glob

# from tensorflow.examples.tutorials.mnist import input_data
from dataset_NEWtf import load_dataset
# Code for saving and restoring the model
import SaveRestoreReset as srr

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

FLAGS = None
def l1_loss(logits, gt):
  valus_diff = tf.abs(tf.subtract(logits, gt))
  L1_loss = tf.reduce_mean(valus_diff)
  return L1_loss


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

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    if debug:
      print("conv1:")
    W_conv1 = weight_variable([5, 5, numFrames, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    if debug:
      print("pool1:")
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    if debug:
      print("conv2:")
    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    if debug:
      print("pool2:")
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 100x100 image
  # is down to 25x25x64 feature maps -- maps this to 65536 features.
  fc1_length = 8192
  with tf.name_scope('fc1'):
    if debug:
      print("fc1:")
    W_fc1 = weight_variable([int(imgSize/4) * int(imgSize/4) * 16, fc1_length])
    b_fc1 = bias_variable([fc1_length])

    h_pool2_flat = tf.reshape(h_pool2, [-1, int(imgSize/4)*int(imgSize/4)*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    if debug:
      print("dropout:")
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    if debug:
      print("fc2:")
    W_fc2 = weight_variable([fc1_length, imgSize*imgSize*maxSources])
    b_fc2 = bias_variable([imgSize*imgSize*maxSources])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="y_conv")
    
  with tf.name_scope('reshape_y'):
    if debug:
      print("reshape_y:")
    y_conv = tf.reshape(y_conv, [-1, imgSize, imgSize, maxSources], name="y_conv")
    # -1 is for the batch size, will be dynamically assigned 
    
  return y_conv, keep_prob

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  if debug:
    print("conv2d:")
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  if debug:
    print("max_pool_2x2:")
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

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
      model_name = 'im64_f8_s4'      
      graph_location = os.path.join(file_path,'graphs','graph_'+model_name)
      ckpt_location = os.path.join(file_path,'checkpoints','ckpt_'+model_name)
      restored_ckpt_name = model_name+'_2018-04-04_0000' # for name mode in restore
      if not os.path.exists(ckpt_location):
        os.makedirs(ckpt_location)
      # Restore params  
      restoreFlag = 0  
      restore_mode = 'last' #last - take last checkpoint, name - get apecific checkpoint by name, best - take checkpoint with best accuracy so far (not supported yet)
      
      # Manage checkpoints log
      log_obj = srr.get_log(ckpt_location, model_name)
      log_obj.write('\n' + ('#' * 50))
      ckpt_start_time = srr.get_time()
      log_obj.write("\ncheckpoint name: %s" % model_name + '_' + ckpt_start_time)
      
      with tf.name_scope('data'):  
          # Import data
          first_sample = 1
          num_samp = 10
          iter_num = num_samp
          dataObj, imgSize, numFrames, maxSources = load_dataset(first_sample,num_samp)
          data_params = [imgSize, numFrames, maxSources]
          print("loaded data with the following params:")
          print("imgSize is:" +str(imgSize))
          print("numFrames is:" +str(numFrames))
          print("maxSources is:" +str(maxSources))
          batch_size = 1
    
      # Create the model
      x = tf.placeholder(tf.float32, [None, imgSize, imgSize, numFrames], name = 'x')
      y_ = tf.placeholder(tf.float32, [None, imgSize, imgSize, maxSources], name = 'y_')
      global_step = tf.Variable(0, name='global_step', trainable=False)
    
      # Build the graph for the deep net
      y_conv, keep_prob = deepnn(x,data_params)
    
      # Define loss and optimizer
      with tf.name_scope('loss'):
#        loss = tf.reduce_mean(tf.losses.mean_squared_error(y_,y_conv))
        loss = l1_loss(y_,y_conv)
        
        
    
      with tf.name_scope('adam_optimizer'):
        lr = 1e-4
        train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    
      with tf.name_scope('accuracy'):
         accuracy = cross_corr(y_conv, y_, batch_size, data_params)
         
      # Create Summaries
      tf.summary.scalar("loss", loss)
      tf.summary.scalar("accuracy", accuracy)
      tf.summary.image('label1', tf.cast(tf.expand_dims(y_[:,:,:,0], axis=-1)*4,tf.uint8))
      tf.summary.image('label2', tf.cast(tf.expand_dims(y_[:,:,:,1], axis=-1)*4,tf.uint8))
      tf.summary.image('result1', tf.cast(tf.expand_dims(y_conv[:,:,:,0], axis=-1)*4,tf.uint8))
      tf.summary.image('result2', tf.cast(tf.expand_dims(y_conv[:,:,:,1], axis=-1)*4,tf.uint8))
      
      
      #tf.summary.image('label1', tf.cast(tf.slice(y_, [0, 0, 0, 0], [1, 64, 64, 1]), tf.uint8))
      #tf.summary.image('label2', tf.cast(tf.slice(y_, [0, 0, 0, 1], [1, 64, 64, 2]), tf.uint8))
      #tf.slice(logits[2], [0, 0, 0, 0], [FLAGS.batch, FLAGS.output_height, FLAGS.output_width, 1]
      #tf.summary.image('result1', tf.cast(tf.slice(y_conv, [0, 0, 0, 0], [1, 64, 64, 1]), tf.uint8))
      #tf.summary.image('result2', tf.cast(tf.slice(y_conv, [0, 0, 0, 1], [1, 64, 64, 2]), tf.uint8))
      # because you have several summaries, we should merge them all
      # into one op to make it easier to manage
      summary_op = tf.summary.merge_all()
    
      with tf.Session() as sess:
          
        sess.run(tf.global_variables_initializer())

        # Create writer objects
        print('Saving graph to: %s' % graph_location)
        files_before = glob.glob(os.path.join(graph_location,'*'))
        train_writer = tf.summary.FileWriter(graph_location, graph=tf.get_default_graph())
        new_file = set(files_before).symmetric_difference(set(glob.glob(os.path.join(graph_location,'*'))))
        log_obj.write("\n"+"Graph file name: %s" % (''.join(new_file)))
        
        if restoreFlag:
            res_name = srr.restore(sess, ckpt_location, restored_ckpt_name, restore_mode)
            log_obj.write("\n"+"restored model name: %s" % res_name)
            log_obj.write("\n"+"samples indices from: %d to %d, with total %d iterations" % (first_sample,first_sample+num_samp, iter_num))
            
        
        for i in range(iter_num):
          if i == 0: print("started training") 
          batch = dataObj.train.next_batch(batch_size)
          _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) #training step
          if i % 10 == 0: #TODO: np.floor(iter_num/10) == 0: 
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            train_writer.add_summary(summary, i)       
        print('finished at global step %s' % sess.run(global_step))
        log_obj.write("\n"+"train accuracy: %s" % accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
        log_obj.write("\n"+"finished: %s" % srr.get_time())
        log_obj.close()
        # Saving checkpoints
        srr.save(sess, ckpt_location, model_name + '_' + ckpt_start_time)
        # Saving graph
        train_writer.close() 
          
    ###########     start test section:
        print({x: batch[0], keep_prob: 0.5})
        print(y_conv)
        for_print = sess.run(y_conv, feed_dict={x: batch[0], keep_prob: 0.5})
        print(batch[1].shape)
#        print (x.name)
#        print (keep_prob.name)
#        for_print= for_print[0, :, :, 1]
#        plt.figure(1)
#        plt.imshow(for_print)
#        y_img = batch[1][0, :, :, 1]
#        plt.figure(2)
#        plt.imshow(y_img)
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
  