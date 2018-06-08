import os

import Train
from dataset_NEWtf import load_dataset

def create_cfg(dataset):
    # Init Config Empty Nodes
    cfg = type('', (), {})
    cfg.arch = type('', (), {})
    cfg.paths = type('', (), {})
    cfg.paths.graphs = type('', (), {})
    cfg.data = type('', (), {})
    cfg.load = type('', (), {})
    cfg.exp = type('', (), {})
    cfg.FLAGS = type('', (), {})
    cfg.restore = type('', (), {})
    
    # FLAGS
    cfg.FLAGS.save_ckpt = False
    cfg.FLAGS.LoadObj = dataset[3]
    
    # Directories    
    file_path = os.path.dirname(os.path.abspath(__file__))    
    cfg.paths.dataset = os.path.join(file_path,'..','DataSimulation','Dataset_' + dataset[0])
    cfg.paths.graphs.base = os.path.join(file_path, 'graphs', dataset[0])  
    cfg.paths.ckpts = os.path.join(file_path, 'checkpoints', dataset[0])  
    # TODO:
    cfg.paths.best_model = 'best_model.keras'

    # Load Data Fields
    cfg.load.first_sample = dataset[1]
    cfg.load.numSamples = dataset[2]
    
    # Data Fields
    dataObj, imgSize, numFrames, maxSources = load_dataset(cfg.load.first_sample, 
                                                           cfg.load.numSamples, 
                                                           cfg.paths.dataset, 
                                                           cfg.FLAGS.LoadObj)
    cfg.data.name = dataset[0]
    cfg.data.maxSources = maxSources
    cfg.data.numFrames= numFrames
    cfg.data.imgSize = imgSize
    cfg.data.obj = dataObj
    
    # Current Experiment    
    # TODO: cfg.exp.mode = mode # validation/train/test
    cfg.exp.batch = 1 
    cfg.exp.epochs = 10 
    
    # Architecture Parameters
    # TODO: cfg.arch.model = model # Deconv/FC...
    cfg.arch.lr = 1e-3
    
    # Restore Model
    cfg.restore.mode = 'last'
    cfg.restore.flag = False
    cfg.restore.model = None
    
    return cfg
    

def config_handler(cfg, param_name, value):

    if not os.path.exists(cfg.paths.graphs.base):
        os.makedirs(cfg.paths.graphs.base)
    if not os.path.exists(os.path.join(cfg.paths.graphs.base, cfg.arch.model)):
        os.makedirs(os.path.join(cfg.paths.graphs.base, cfg.arch.model))
    if not os.path.exists(os.path.join(cfg.paths.graphs.base, cfg.arch.model, param_name)):
        os.makedirs(os.path.join(cfg.paths.graphs.base, cfg.arch.model, param_name))
    cfg.paths.graphs.value = os.path.join(cfg.paths.graphs.base, cfg.arch.model, param_name, str(value))
    if not os.path.exists(cfg.paths.graphs.value):
        os.makedirs(cfg.paths.graphs.value)

    
    if param_name == 'arch.lr':
        cfg.arch.lr = value
        
#    elif param_name == 'lr':
#    elif param_name == 'lr':    
#    elif param_name == 'lr':    
    print(cfg.paths.graphs.value)
    return cfg
    
    
def execute_exp(cfg):
    Train.main(cfg)
#    if (cfg.exp_info.run_mode == 'validation'):
#        param_name = cfg.exp_info.valid_param
#        method_to_call = getattr(cfg, param_name)
#        for i in cfg.exp_info.valid_values:
#            method_to_call() = i # maybe method_to_call(i)
#            train(cfg)


    
    
    
    
    
    
    
    
    
    
    
    
    