# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 09:56:11 2018

@author: sorrav
"""

#%%
# Load DataSet
from dataset_NEWtf import load_dataset
# Code for saving and restoring the model
import SaveRestoreReset as srr
# Layers for DeepNN Model
import Layers as layers

import sys
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


def deepnn(x,data_params):
  """deepnn builds the graph for a deep net for seperating emitters.
  Args:
    data_params: [imgSize, numFrames, maxSources]  
    x: an input tensor of shape 64 x 64 x numFrames
  Returns:
    y_conv is a tensor of shape 64 x 64 x maxSources 
  """
  
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - it would be 1 for grayscale
  # 3 for an RGB image, 4 for RGBA, numFrames for a movie.    
  maxSources = data_params[2]
  numFrames = data_params[1]
  imgSize = data_params[0]
  with tf.name_scope('reshape_x'):
    x_image = tf.reshape(x, [-1, imgSize, imgSize, numFrames])
    # -1 is for the batch size, will be dynamically assigned 

  # First Convolutional Layer + maxPooling with argmax
  with tf.name_scope('conv1'):
    conv1_1 = layers.conv(x_image, [5, 5, numFrames, 32], 'conv1_1')  
    conv1_2 = layers.conv(conv1_1, [5, 5, 32, 32], 'conv1_2')  
    h_pool1, argmax1 = layers.max_pool_argmax(conv1_2, 2, 'pool1')
    # Maps to 32x32x32 feature map

  # Second Convolutional Layer + maxPooling with argmax
  with tf.name_scope('conv2'):
    conv2_1 = layers.conv(h_pool1, [3, 3, 32, 64], 'conv2_1')  
    conv2_2 = layers.conv(conv2_1, [3, 3, 64, 64], 'conv2_2')  
    h_pool2, argmax2 = layers.max_pool_argmax(conv2_2, 2, 'pool2')
    # Maps to 16x16x64 feature map
    
  # Third Convolutional Layer + maxPooling with argmax
  with tf.name_scope('conv3'):
    conv3_1 = layers.conv(h_pool2, [3, 3, 64, 128], 'conv3_1')  
    conv3_2 = layers.conv(conv3_1, [3, 3, 128, 128], 'conv3_2')  
    conv3_3 = layers.conv(conv3_2, [3, 3, 128, 128], 'conv3_3')  
    h_pool3, argmax3 = layers.max_pool_argmax(conv3_3, 2, 'pool3')
    # Maps to 8x8x128 feature map
    
#  # Dropout - controls the complexity of the model, prevents co-adaptation of
#  # features.
#  with tf.name_scope('dropout'):
#    if debug:
#      print("dropout:")
#    keep_prob = tf.placeholder(tf.float32)
#    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # First Deconvolutional Layer + Unpooling with argmax
  with tf.name_scope('deconv1'):
    h_unpool1 = layers.unpool_argmax(h_pool3, argmax3, 'unpool1')  
    deconv1_1 = layers.deconv(h_unpool1, [3, 3, 128, 128], 'deconv1_1')
    deconv1_2 = layers.deconv(deconv1_1, [3, 3, 128, 128], 'deconv1_2')
    deconv1_3 = layers.deconv(deconv1_2, [3, 3, 64, 128], 'deconv1_3')
    # Maps to 16x16x64
          
  # Second Deconvolutional Layer + Unpooling with argmax
  with tf.name_scope('deconv2'):
    h_unpool2 = layers.unpool_argmax(deconv1_3, argmax2, 'unpool2')  
    deconv2_1 = layers.deconv(h_unpool2, [3, 3, 64, 64], 'deconv2_1')
    deconv2_2 = layers.deconv(deconv2_1, [3, 3, 32, 64], 'deconv2_2')
    # Maps to 32x32x32
    
  # Third Deconvolutional Layer + Unpooling with argmax
  with tf.name_scope('deconv3'):
    h_unpool3 = layers.unpool_argmax(deconv2_2, argmax1, 'unpool3')  
    deconv3_1 = layers.deconv(h_unpool3, [3, 3, 32, 32], 'deconv3_1')
    deconv3_2 = layers.deconv(deconv3_1, [3, 3, maxSources, 32], 'deconv3_2')
    deconv3_3 = layers.deconv(deconv3_2, [3, 3, maxSources, 32], 'deconv3_3', 'linear')
    
  with tf.name_scope('reshape_y'):
    y_conv = tf.reshape(deconv3_3, [-1, imgSize, imgSize, maxSources])
    # -1 is for the batch size, will be dynamically assigned 
    
  return y_conv

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
      y_conv = deepnn(x,data_params)
    
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
          _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1]})
          if i % 5 == 0: 
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            train_writer.add_summary(summary, i)          
         
        log_obj.write("\ntrain accuracy: %s" % accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))
        log_obj.write("\nfinished: %s" % srr.get_time())
        log_obj.close()
        # Saving checkpoints
        srr.save(sess, ckpt_location, model_name + '_' + ckpt_start_time)
        # Saving graph
        train_writer.close() 
          
    ###########     start test section:
        for_print = sess.run(y_conv, feed_dict={x: batch[0]})
        for_print= for_print[0, :, :, 1]
        plt.figure(1)
        plt.imshow(for_print)
        y_img = batch[1][0, :, :, 1]
        plt.figure(2)
        plt.imshow(y_img)
    ###########      end test section:
    
        print('test accuracy %g' % accuracy.eval(feed_dict={
                x: dataObj.test.features, y_: dataObj.test.labels}))
                
  except Exception:
      log_obj.close()
      train_writer.close() 

if __name__ == '__main__':
  debug= False
  if debug:
    print("started")
  tf.app.run(main=main, argv=[sys.argv[0]])
  