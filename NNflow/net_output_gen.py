# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:38:07 2018

@author: Alexfi
"""
import tensorflow as tf

def net_output_gen(graph_path, ckpt_path, x_):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(graph_path)
        new_saver.restore(sess, ckpt_path)
        out = sess.run(y_conv, feed_dict={x: x_, keep_prob: 0.5})

    return out