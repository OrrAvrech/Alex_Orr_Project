# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:19:16 2018

@author: afinkels
"""
import UserConfig as user

model_name = 'Deconv'
run_mode = 'Experiment'
dataset_name = 'im64_f8_s2'
firstSample = 1
numSamples = 10
LoadObj = False
dataset_params = [dataset_name, firstSample, numSamples, LoadObj]

cfg = user.create_cfg(dataset_params, run_mode, model_name)
################
## fill out

# Data Fields

# Directories    

# Load Data Fields

# Current Experiment    
cfg.exp.batch = 1
cfg.exp.epochs = 500

# Architecture Parameters
cfg.arch.lr = 1e-3
###############

##########
## Execute
##########
param_name = 'arch.lr'
param_vals = [1e-3]
for i in param_vals:
    user.config_handler(cfg, param_name, i)
    user.execute_exp(cfg)
#    print(cfg.arch.lr)
#user.config_handler(cfg, 'lr', 0.2)
    