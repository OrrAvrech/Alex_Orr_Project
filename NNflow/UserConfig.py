
import os

import Train

def create_cfg(dataset, mode, model):
    # Init Config Empty Nodes
    cfg_node = type('', (), {}) #pointer to struct type
    cfg = cfg_node
    cfg.arch = cfg_node
    cfg.paths = cfg_node
    cfg.paths.graphs = cfg_node
    cfg.data = cfg_node
    cfg.load = cfg_node
    cfg.exp = cfg_node

    # Data Fields
    cfg.data.name = dataset
    cfg.data.maxSources = None 
    cfg.data.numFrames= None 
    cfg.data.imgSize = None 
    
    # Directories    
    file_path = os.path.dirname(os.path.abspath(__file__))    
    cfg.paths.graphs.base = os.path.join(file_path, 'graphs', dataset)  
    
    # Load Data Fields
    cfg.load.first_sample = 1 
    cfg.load.numSamples = 10
    
    # Current Experiment    
    cfg.exp.mode = mode # validation/train/test
    cfg.exp.batch = 1 
    cfg.exp.epochs = 100 
    
    # Architecture Parameters
    cfg.arch.model = model # Deconv/FC...
    cfg.arch.lr = 1e-3
    
    return cfg
    

def config_handler(cfg, param_name, value):

    if not os.path.exists(cfg.paths.graphs.base):
        os.makedirs(cfg.paths.graphs.base)
    if not os.path.exists(os.path.join(cfg.paths.graphs.base, cfg.arch.model)):
        os.makedirs(os.path.join(cfg.paths.graphs.base, cfg.arch.model))
    if not os.path.exists(os.path.join(cfg.paths.graphs.base, cfg.arch.model, param_name)):
        os.makedirs(os.path.join(cfg.paths.graphs.base, cfg.arch.model, param_name))
    if not os.path.exists(os.path.join(cfg.paths.graphs.base, cfg.arch.model, param_name, str(value))):
        cfg.paths.graphs.value = os.makedirs(os.path.join(cfg.paths.graphs.base, cfg.arch.model, param_name, str(value)))
    
    if param_name == 'arch.lr':
        cfg.arch.lr = value
        
#    elif param_name == 'lr':
#    elif param_name == 'lr':    
#    elif param_name == 'lr':    
    
    return cfg
    
    
def execute_exp(cfg):
    Train.main(cfg)
#    if (cfg.exp_info.run_mode == 'validation'):
#        param_name = cfg.exp_info.valid_param
#        method_to_call = getattr(cfg, param_name)
#        for i in cfg.exp_info.valid_values:
#            method_to_call() = i # maybe method_to_call(i)
#            train(cfg)


    
    
    
    
    
    
    
    
    
    
    
    
    