# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:21:47 2018

@author: sorrav
"""
#
#class Config:
#    
#    SECRET_KEY = 'secret-key-of-myapp'
#    ADMIN_NAME = 'administrator'
#
#    AWS_DEFAULT_REGION = 'ap-northeast-2'
#    
#    STATIC_PREFIX_PATH = 'static'
#    ALLOWED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'gif']
#    MAX_IMAGE_SIZE = 5242880 # 5MB
#
#    
#class DevelopmentConfig(Config):
#    DEBUG = True
#    
#    AWS_ACCESS_KEY_ID = 'aws-access-key-for-dev'
#    AWS_SECERT_ACCESS_KEY = 'aws-secret-access-key-for-dev'
#    AWS_S3_BUCKET_NAME = 'aws-s3-bucket-name-for-dev'
#    
#    DATABASE_URI = 'database-uri-for-dev'
#    
#
#
#class Exp_Config:
#    ##deault vals:
#    parent_path = None
#    graph_path = None
#    exp_path = None
#    maxSources = None
#    numFrames = None
#    numSamples = None
#    imgSize = None
#    
#    _CONFIG = {
#    'paths': {'parent_path': parent_path, 'graph_path': graph_path, 'exp_path': exp_path},
#    'dataParams': {'maxSources': maxSources, 'numFrames': numFrames, 'numSamples': numSamples, 'imgSize': imgSize],
#    'flage': 'value',
#    }
#    
#    
#    
def create_cfg(mode, model, num_samples):
    cfg_node = type('', (), {}) #pointer to strruct type
    cfg = cfg_node
    cfg.arch_info = cfg_node
    cfg.paths = cfg_node
    cfg.data_params = cfg_node
    cfg.load_info = cfg_node
    cfg.exp_info = cfg_node
    cfg.data_params.maxSources = None 
    cfg.data_params.numFrames= None 
    cfg.data_params.imgSize = None 
    cfg.load_info.first_sample = 1 #default
    cfg.load_info.numSamples = num_samples
    cfg.exp_info.epochs = 100 #default
    cfg.arch_info.model = model
    cfg.paths.arch_path = 'arch_path'
    cfg.exp_info.mode = 'mode' #default
    cfg.exp_info.batch = 1 #default
    
    
    return cfg

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    