import os

import UserConfig as user
import Train_keras as train

#%% Create Configuration Object

def log_dir_name(isExperiment, dir_name):

    # The dir-name for the TensorBoard log-dir.
    if isExperiment == True:
        log_dir = os.path.join('Experiments', dir_name)
    else:
        log_dir = os.path.join('Optimized', dir_name)
        
    return log_dir
    
# =============================================================================
# Dataset used for training and evaluation: dataset_name
# =============================================================================
dataset_name = 'im64_f8_s2'

# =============================================================================
# Loading a data object from the given dataset: Load Obj
# =============================================================================
# False - Usage:
#               data objects do not exist 
#               using a portion of the dataset for execution
#               testing code
#               debugging   
# True - Usage:
#               fast loading of the entire dataset
LoadObj = False

# TODO: Model


cfg = user.create_cfg(dataset_name, LoadObj)
# if LoadObj is False -  denote the range of files to load (or leave as default):
#cfg.load.first_sample = 1  # (default = 1)
#cfg.load.numSamples   = 15 # (default = 15)

#%% Experiments
cfg.paths.log_dir = os.path.join(cfg.paths.log_home, log_dir_name(True, 'baseline'))

accuracy = train.fit_model(cfg)


#%% hyperopt