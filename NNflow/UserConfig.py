import os
import Train_keras

def init():
    empty_node = type('', (), {})
    return empty_node

def create_cfg(dataset_params, model, run_mode):
    # Init Config Empty Nodes
    cfg = init()
    
    # FLAGS
    cfg.FLAGS = init()
    
    # Data Fields
    cfg.data = init()
    cfg.data.maxSources = dataset_params[2]
    cfg.data.numFrames= dataset_params[1]
    cfg.data.imgSize = dataset_params[0]
    cfg.data.name = 'im'+str(dataset_params[0])+'_f'+str(dataset_params[1])+'_s'+str(dataset_params[2])
    
    # Directories    
    cfg.paths                   = init()
    file_path                   = os.path.dirname(os.path.abspath(__file__))    
    cfg.paths.dataset           = os.path.join(file_path,'..','DataSimulation','Dataset_' + cfg.data.name)
    cfg.paths.best_models       = os.path.join(file_path,'best_models', cfg.data.name)
    cfg.paths.summaries_dataset = os.path.join(file_path, 'summaries', cfg.data.name)
    cfg.paths.summaries_model   = os.path.join(cfg.paths.summaries_dataset, model)
    cfg.paths.summaries_runBase = os.path.join(cfg.paths.summaries_model, run_mode)
    cfg.paths.summaries_current = cfg.paths.summaries_runBase
    cfg.paths.model             = os.path.join(cfg.paths.best_models, 'best_model.keras')

    # Load Data Fields
    cfg.load = init()
    cfg.load.first_sample = 1
    cfg.load.numSamples = 15
    
#    dataObj, imgSize, numFrames, maxSources = load_dataset(cfg.load.first_sample, 
#                                                           cfg.load.numSamples, 
#                                                           cfg.paths.dataset, 
#                                                           cfg.FLAGS.LoadObj)
#    cfg.data.obj = dataObj    
    
    # Current Experiment    
    cfg.exp = init()
    cfg.exp.batch = 1
    cfg.exp.epochs = 10
    
    # Architecture Parameters
    cfg.arch = init()
    cfg.arch.model = model # Deconv/FC...
    cfg.arch.lr = 1e-3
    
    # Restore Model
#    cfg.restore = init()
#    cfg.restore.mode = 'last'
#    cfg.restore.flag = False
#    cfg.restore.model = None
    
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

    