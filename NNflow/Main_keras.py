import os

import UserConfig as user
import Train_keras as train

dataset_name = 'im64_f8_s4'
firstSample = 1
numSamples = 100
LoadObj = False
dataset_params = [dataset_name, firstSample, numSamples, LoadObj]
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

#%% Create Configuration Object

#def log_dir_name(isExperiment, dir_name):
#
#    # The dir-name for the TensorBoard log-dir.
#    if isExperiment == True:
#        log_dir = os.path.join('Experiments', dir_name)
#    else:
#        log_dir = os.path.join('Optimized', dir_name)
#        
#    return log_dir
    
# =============================================================================
# Dataset used for training and evaluation: dataset_name
# =============================================================================
dataset_name = 'im64_f8_s2'

# =============================================================================
# Run Mode
# =============================================================================
# Regular   : single run with current parameters
# Experiment: multiple run over a set of tested parameters
# Optimize  : execute a hyperparam optimization algorithm 
#run_mode = 'Experiment'
run_mode = 'Optimize'

# =============================================================================
# Model
# =============================================================================
model = 'DeconvN'

# =============================================================================
# Loading a data object from the given dataset: Load Obj
# =============================================================================
# False - Usage:
#               data objects do not exist 
#               testing code
#               debugging   
# True - Usage:
#               fast loading of the entire dataset
#LoadObj = False

cfg = user.create_cfg(dataset_name, model, run_mode)
# if LoadObj is False -  denote the range of files to load (or leave as default):
#cfg.load.first_sample = 1  # (default = 1)
#cfg.load.numSamples   = 15 # (default = 15)
cfg.exp.batch         = 2  # (default = 1)   
#cfg.exp.epochs        = 10 # (default = 10)

#%% Regular
if run_mode == 'Regular':    
    accuracy = train.fit_model(cfg)

#%% Experiments
if run_mode == 'Experiment':
    param_name = 'arch.lr'
    param_vals = [1e-3, 2e-3]
    for i in param_vals:
        user.config_handler(cfg, param_name, i)
        user.execute_exp(cfg)

#%% Optimize
if run_mode == 'Optimize':
    # Hyperparams
    
    dim_learning_rate = Real(low=2.3e-3, high=2.5e-3, prior='log-uniform', name='learning_rate')
    dim_num_conv_Bulks = Integer(low=1, high=5, name='num_conv_Bulks')
    #dim_num_conv_Layers = Integer(low=1, high=5, name='num_conv_Layers')
    dim_kernel_size = Categorical(categories=[3, 5], name='kernel_size')
    dim_activation = Categorical(categories=['relu', 'linear'], name='activation')
    default_parameters = [1e-5, 1, 16, 'relu']
    #cfg.hyper.param = categorical (list)
#    cfg.hyper.param = categorical (one_param)
#    accuracy = train.fit_model()
#    dim_batch_size = Integer(low=1, high=4, name='batch_size') #TODO: decide if needed
#    dim_num_epochs = Integer(low=5, high=50, name='num_epochs') #TODO: decide if needed
    dimensions = [cfg, dim_learning_rate, dim_num_conv_Bulks, dim_kernel_size, dim_activation]
    gp_minimize(func=train.fit_model,
                dimensions=dimensions,
                acq_func='EI', # Expected Improvement.
                n_calls=40,
                x0=default_parameters)
