from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

import UserConfig as user
import Train_keras as train


#%% Create Configuration Object
    
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


cfg = user.create_cfg(dataset_name, model, run_mode)
# if LoadObj is False -  denote the range of files to load (or leave as default):
#cfg.load.first_sample = 1  # (default = 1)
#cfg.load.numSamples   = 5000 # (default = 15)
cfg.exp.batch         = 2  # (default = 1)   
cfg.exp.epochs        = 5 # (default = 10)

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
    dim_kernel_size = Categorical(categories=[3, 5], name='kernel_size')
    dim_activation = Categorical(categories=['relu', 'linear'], name='activation')
    dim_cfg = Categorical(categories=[cfg], name='cfg')
    default_parameters = [cfg, 2.5e-3, 1, 3, 'relu']
    
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
    print(search_result)