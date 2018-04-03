# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:46:36 2018

@author: sorrav
"""

import tensorflow as tf

def save(sess, ckpt_location, checkpoint_file):
    saver = tf.train.Saver()
    save_path = saver.save(sess, ckpt_location + checkpoint_file)
    print('Model saved in path: %s' % save_path)

def restore(sess, ckpt_location, checkpoint_file):
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_location + checkpoint_file)
    print("Model restored")
#    print(sess.run(tf.all_variables()))

def reset():
    tf.reset_default_graph()
    
def saveGraph(graph_location):
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)    
    train_writer.add_graph(tf.get_default_graph())