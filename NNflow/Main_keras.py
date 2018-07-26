import UserConfig as user
import Train_keras as train

dataset_name = 'im64_f8_s4'
firstSample = 1
numSamples = 100
LoadObj = False
dataset_params = [dataset_name, firstSample, numSamples, LoadObj]

cfg = user.create_cfg(dataset_params)

# Hyperparams
#cfg.hyper.param = categorical (list)
cfg.hyper.param = categorical (one_param)

accuracy = train.fit_model()
