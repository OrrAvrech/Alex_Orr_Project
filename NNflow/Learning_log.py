# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:30:44 2018

@author: sorrav
"""

import datetime

def get_log(path_to_dir, model_name):
    log_file_name = path_to_dir + model_name + '.txt'
    log_obj = open(log_file_name, 'a')
    print("create log: %s" % log_file_name)
    return log_obj

#def write_to(log_file):
    
def get_time():    
    time_stamp = '{date:%Y-%m-%d_%H%M}'.format( date=datetime.datetime.now())
    return time_stamp
    