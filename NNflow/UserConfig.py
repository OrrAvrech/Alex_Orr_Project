import os
import Train_keras

from dataset_NEWtf import load_dataset

def init():
    empty_node = type('', (), {})
    return empty_node

def create_cfg(dataset_name, model, run_mode):
    # Init Config Empty Nodes
    cfg = init()
    
    # FLAGS
    cfg.FLAGS = init()
    cfg.FLAGS.LoadObj = False
#    cfg.FLAGS.save_ckpt = False
    
    # Directories    
    cfg.paths                   = init()
    file_path                   = os.path.dirname(os.path.abspath(__file__))    
    cfg.paths.dataset           = os.path.join(file_path,'..','DataSimulation','Dataset_' + dataset_name)
    cfg.paths.summaries_dataset = os.path.join(file_path, 'summaries', dataset_name)
    cfg.paths.summaries_model   = os.path.join(cfg.paths.summaries_dataset, model)
    cfg.paths.summaries_runBase = os.path.join(cfg.paths.summaries_model, run_mode)
    cfg.paths.summaries_current = cfg.paths.summaries_runBase
    cfg.paths.model             = 'best_model.keras'
#    cfg.paths.ckpts = os.path.join(file_path, 'checkpoints', dataset[0])  

    # Load Data Fields
    cfg.load = init()
    cfg.load.first_sample = 1
    cfg.load.numSamples = 5000
    
    # Data Fields
    dataObj, imgSize, numFrames, maxSources = load_dataset(cfg.load.first_sample, 
                                                           cfg.load.numSamples, 
                                                           cfg.paths.dataset, 
                                                           cfg.FLAGS.LoadObj)
    cfg.data = init()
    cfg.data.name = dataset_name
    cfg.data.maxSources = maxSources
    cfg.data.numFrames= numFrames
    cfg.data.imgSize = imgSize
    cfg.data.obj = dataObj
    
    # Current Experiment    
    cfg.exp = init()
    cfg.exp.batch = 1
    cfg.exp.epochs = 10
    
    # Architecture Parameters
    cfg.arch = init()
    cfg.arch.model = model # Deconv/FC...
    cfg.arch.lr = 1e-3
    
    # Restore Model
    cfg.restore = init()
    cfg.restore.mode = 'last'
    cfg.restore.flag = False
    cfg.restore.model = None
    
    # Hyperparams
        
    
    return cfg

def config_handler(cfg, param_name, value):

    if not os.path.exists(os.path.join(cfg.paths.summaries_runBase, param_name)):
        os.makedirs(os.path.join(cfg.paths.summaries_runBase, param_name))
    cfg.paths.summaries_current = os.path.join(cfg.paths.summaries_runBase, param_name, str(value))
    if not os.path.exists(cfg.paths.summaries_current):
        os.makedirs(cfg.paths.summaries_current)

    if param_name == 'arch.lr':
        cfg.arch.lr = value
        
#    elif param_name == 'lr':
#    elif param_name == 'lr':    
#    elif param_name == 'lr':    
    return cfg
    
def execute_exp(cfg):
    Train_keras.fit_model(cfg)

    