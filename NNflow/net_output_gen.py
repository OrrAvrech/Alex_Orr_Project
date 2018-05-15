# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:38:07 2018

@author: Alexfi
"""
import tensorflow as tf
import SaveRestoreReset as srr
import os


def net_output_gen(out_tens_name, feed_tens_names, feed_values, ckpt_path, ckpt_mode):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        ckpt_path_split = os.path.split(ckpt_path)
        srr.restore(sess, ckpt_path_split[0], ckpt_path_split[1], ckpt_mode, new_saver)
        graph = tf.get_default_graph()
        out_tens = graph.get_tensor_by_name(out_tens_name)
        feed_keys = [graph.get_tensor_by_name(tens_name) for tens_name in feed_tens_names]
        feed_dict = {}
        for i in range(len(feed_keys)):
            feed_dict[feed_keys[i]] = feed_values[i] 
        out = sess.run(out_tens, feed_dict=feed_dict)
    return out