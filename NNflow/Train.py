# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 00:04:29 2018

@author: sorrav
"""

# Load DataSet
from dataset_NEWtf import load_dataset
# Code for saving and restoring the model
import SaveRestoreReset as srr
# Import Models
import Models as models
# Handle Summaries
import SummaryHandler as summariz

import sys
import glob
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

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

def l1_loss(logits, gt):
  valus_diff = tf.abs(tf.subtract(logits, gt))
  L1_loss = tf.reduce_mean(valus_diff)
  return L1_loss

def main(_):
  # Save Graph and Checkpoints
  srr.reset()
  file_path = os.path.dirname(os.path.abspath(__file__))
  graph_location = os.path.join(file_path,'graphs','graph_im64_f8_s2')
  ckpt_location = os.path.join(file_path,'checkpoints','ckpt_im64_f8_s2')
  model_name = 'im64_f8_s2'
  arch_name = 'CNN_deconv'
  restored_ckpt_name = 'im64_f8_s4_2018-04-04_1615' # for name mode in restore
  if not os.path.exists(ckpt_location):
    os.makedirs(ckpt_location)
  # Restore params  
  restoreFlag = 0  
  restore_mode = 'last' #last - take last checkpoint, name - get apecific checkpoint by name, best - take checkpoint with best accuracy so far (not supported yet)
  
  # Manage checkpoints log
  log_obj = srr.get_log(ckpt_location, model_name+ '_' +arch_name)
  log_obj.write('\n' + ('#' * 50))
  ckpt_start_time = srr.get_time()
  log_obj.write("\ncheckpoint name: %s" % model_name + '_' + ckpt_start_time)
  
  with tf.name_scope('data'):  
      # Import data
      first_sample = 1
      num_samp = 10#00
      epochs = 10#00
      iter_num = num_samp*epochs
      dataObj, imgSize, numFrames, maxSources = load_dataset(first_sample,num_samp)
      data_params = [imgSize, numFrames, maxSources]
      print("loaded data with the following params:")
      print("imgSize is:" +str(imgSize))
      print("numFrames is:" +str(numFrames))
      print("maxSources is:" +str(maxSources))
      batch_size = 1

  # Create the model
  x = tf.placeholder(tf.float32, [None, imgSize, imgSize, numFrames], name='x')
  y_ = tf.placeholder(tf.float32, [None, imgSize, imgSize, maxSources], name='y_')
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # Build the graph for the deep net
  y_conv, keep_prob = models.ConvFCN(x,data_params)

  # Define loss and optimizer
  with tf.name_scope('loss'):
     loss = l1_loss(y_,y_conv) 

  with tf.name_scope('adam_optimizer'):
    lr = 1e-3
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

  with tf.name_scope('accuracy'):
     accuracy = cross_corr(y_conv, y_, batch_size, data_params)
     
  # Define and Merge Summaries
  summariz.define_summaries(loss, accuracy, y_, y_conv)
  summary_op = tf.summary.merge_all()
     
  with tf.Session() as sess:
    print('Initialize global variables')
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
      batch_test = dataObj.test.next_batch(batch_size)
      _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) #training step
      if i % np.floor(iter_num/10) == 0: 
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        train_writer.add_summary(summary, i)       
    print('finished at global step %s' % sess.run(global_step))
    log_obj.write("\n"+"train accuracy: %s" % accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
    log_obj.write("\n"+"finished: %s" % srr.get_time())
    log_obj.close()
    # Saving checkpoints
    srr.save(sess, ckpt_location, model_name + '_' + ckpt_start_time)
    train_writer.close()
      
###########     start test section:
    for_print = sess.run(y_conv, feed_dict={x: batch[0], keep_prob: 0.5})
    for_print= for_print[0, :, :, 1]
    plt.figure(1)
    plt.imshow(for_print)
    y_img = batch[1][0, :, :, 1]
    plt.figure(2)
    plt.imshow(y_img)
    
    for_print = sess.run(y_conv, feed_dict={x: batch_test[0], keep_prob: 0.5})
    for_print= for_print[0, :, :, 1]
    plt.figure(3)
    plt.imshow(for_print)
    y_img = batch_test[1][0, :, :, 1]
    plt.figure(4)
    plt.imshow(y_img)        
###########      end test section:

    print('test accuracy %g' % accuracy.eval(feed_dict={
            x: dataObj.test.features, y_: dataObj.test.labels, keep_prob: 1.0}))
 
 
if __name__ == '__main__':
  debug=True
  if debug:
    print("start")
  try:  
      tf.app.run(main=main, argv=[sys.argv[0]])                
  except SystemExit:
      print("end")  

