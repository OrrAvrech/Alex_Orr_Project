import os

from dataset_NEWtf import load_dataset

def init():
    empty_node = type('', (), {})
    return empty_node

def create_cfg(dataset_name, LoadObj):
    # Init Config Empty Nodes
    cfg = init()
    
    # FLAGS
    cfg.FLAGS = init()
    cfg.FLAGS.LoadObj = LoadObj
#    cfg.FLAGS.save_ckpt = False
    
    # Directories    
    cfg.paths = init()
    file_path = os.path.dirname(os.path.abspath(__file__))    
    cfg.paths.dataset = os.path.join(file_path,'..','DataSimulation','Dataset_' + dataset_name)
    cfg.paths.log_home = os.path.join(file_path,'logs',dataset_name)
    cfg.paths.log_dir = os.path.join(file_path,'logs',dataset_name,'Experiments')
    cfg.paths.model = 'best_model.keras'
#    cfg.paths.graphs.base = os.path.join(file_path, 'graphs', dataset[0])  
#    cfg.paths.ckpts = os.path.join(file_path, 'checkpoints', dataset[0])  

    # Load Data Fields
    cfg.load = init()
    cfg.load.first_sample = 1
    cfg.load.numSamples = 15
    
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
    cfg.exp.epochs = 100
    
    # Architecture Parameters
    cfg.arch = init()
    # TODO: cfg.arch.model = model # Deconv/FC...
    cfg.arch.lr = 1e-3
    
    # Restore Model
    cfg.restore = init()
    cfg.restore.mode = 'last'
    cfg.restore.flag = False
    cfg.restore.model = None
    
    return cfg
    