# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:46:36 2018

@author: sorrav
"""

import tensorflow as tf
import os
import datetime

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

def get_log(path_to_dir, model_name):
    log_file_name = os.path.join(path_to_dir , model_name + '.txt')
    log_obj = open(log_file_name, 'a')
    print("create log: %s" % log_file_name)
    return log_obj

#def write_to(log_file):
    
def get_time():    
    time_stamp = '{date:%Y-%m-%d_%H%M}'.format( date=datetime.datetime.now())
    return time_stamp