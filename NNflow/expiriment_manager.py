# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:19:16 2018

@author: afinkels
"""
import UserConfig as user

model_name = 'Deconv'
run_mode = 'validation'
dataset_name = 'im64_f16_s4'

cfg = user.create_cfg(dataset_name, run_mode, model_name)
################
## fill out

# Data Fields
cfg.data.maxSources = None 
cfg.data.numFrames= None 
cfg.data.imgSize = None 

# Directories    

# Load Data Fields
cfg.load.first_sample = 1  

# Current Experiment    
cfg.exp.batch = 1 
cfg.exp.epochs = 100 

# Architecture Parameters
cfg.arch.lr = 1e-3

###############

##########
## Execute
##########
param_name = 'arch.lr'
param_vals = [1.2, 1.3]
for i in param_vals:
    user.config_handler(cfg, param_name, i)
    user.execute_exp(cfg)
#    print(cfg.arch.lr)
#user.config_handler(cfg, 'lr', 0.2)
    