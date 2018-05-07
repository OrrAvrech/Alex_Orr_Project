# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:19:16 2018

@author: afinkels
"""
from UserConfig import create_cfg


def execute_exp(cfg):
    if (cfg.exp_info.run_mode == 'validation'):
        param_name = cfg.exp_info.valid_param
        method_to_call = getattr(cfg, param_name)
        for i in cfg.exp_info.valid_values:
            method_to_call() = i # maybe method_to_call(i)
            train(cfg)





model_name = 'name'
run_mode = 'mode'
num_samples = 1000

cfg = create_cfg(run_mode, model_name, num_samples)



execute_exp(cfg)