# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:46:36 2018

@author: sorrav
"""

import tensorflow as tf
import os

def save(sess, ckpt_location, checkpoint_file):
    saver = tf.train.Saver()
    save_path = saver.save(sess, os.path.join(ckpt_location,checkpoint_file))
    print('Model saved in path: %s' % save_path)

def restore(sess, ckpt_location, checkpoint_file, mode):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(ckpt_location , 'checkpoint')))
    if not ckpt:
        print("Nothing to restore from")
        return "None"
    if mode == 'last':
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_location))
        res_name = os.path.split(tf.train.latest_checkpoint(ckpt_location))[1]
    if mode == 'name':    
        if not os.path.exists(os.path.join(ckpt_location, checkpoint_file + '.meta')):
            print("Invalid checkpoint name")
            exit()
        saver.restore(sess, os.path.join(ckpt_location, checkpoint_file))
        res_name = checkpoint_file
    print("Model %s restored" % res_name)
    return res_name
    
def reset():
    tf.reset_default_graph()   