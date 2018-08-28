from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

import UserConfig as user
import Train_keras as train
from dataset_NEWtf import load_dataset

#%% Create Configuration Object
    
# =============================================================================
# Dataset parameters used for training and evaluation: dataset_params
# =============================================================================
imgSize = 64
numFrames = 64
maxSources = 32
dataset_params = [imgSize, numFrames, maxSources]

# =============================================================================
# Run Mode
# =============================================================================
# Regular   : single run with current parameters
# Experiment: multiple run over a set of tested parameters
# Optimize  : execute a hyperparam optimization algorithm 
run_mode = 'Regular'

# =============================================================================
# Model
# =============================================================================
model = 'DeconvN'

#==============================================================================
# Data
#==============================================================================
cfg = user.create_cfg(dataset_params, model, run_mode)
cfg.load.first_sample = 1  # (default = 1) #not needed if Loadobj=True
cfg.load.numSamples   = 15# (default = 15) #not needed if Loadobj=True
LoadObj = True
SaveObj = False

cfg.data.obj,_,_,_ =  load_dataset(cfg.load.first_sample, cfg.load.numSamples, cfg.paths.dataset, LoadObj, SaveObj)

cfg.exp.batch         = 20  # (default = 1)
cfg.exp.epochs        = 1000 # (default = 10)

#%% Regular
if run_mode == 'Regular':
    cfg.restore.flag = True
    learning_rate = 3.078999840939997e-05
    num_conv_Bulks = 3
    kernel_size = 5
    activation = 'relu'
    accuracy = train.fit_model(cfg, learning_rate, num_conv_Bulks, kernel_size, activation)

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
    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_num_conv_Bulks = Integer(low=1, high=5, name='num_conv_Bulks')
    dim_num_conv_Layers = Integer(low=1, high=5, name='num_conv_Layers')
    dim_kernel_size = Categorical(categories=[3, 5], name='kernel_size')

    dim_activation = Categorical(categories=['sigmoid', 'linear', 'relu'], name='activation')
    dim_cfg = Categorical(categories=[cfg], name='cfg')
    default_parameters = [cfg, 2e-3, 1, 3, 'sigmoid']
    
#    dim_batch_size = Integer(low=1, high=4, name='batch_size') #TODO: decide if needed
#    dim_num_epochs = Integer(low=5, high=50, name='num_epochs') #TODO: decide if needed
    
    dimensions = [dim_cfg, dim_learning_rate, dim_num_conv_Bulks, dim_kernel_size, dim_activation]
    
    @use_named_args(dimensions = dimensions)
    def train_wrapper(cfg, learning_rate, num_conv_Bulks, kernel_size, activation):
        return train.fit_model(cfg, learning_rate, num_conv_Bulks, kernel_size, activation)
    
    search_result = gp_minimize(func=train_wrapper,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=11,
                                x0=default_parameters)