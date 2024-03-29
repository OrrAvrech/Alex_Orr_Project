# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:08:40 2018

@author: Alexfi
"""

import net_output_gen as nog
import os
from dataset_NEWtf import load_dataset
import matplotlib.pyplot as plt
import tensorflow as tf


file_path = os.path.dirname(os.path.abspath(__file__))
model_name = 'im64_f8_s2'
ckpt_name = model_name+'_2018-05-11_1356'
ckpt_path = os.path.join(file_path,'checkpoints','ckpt_'+model_name,ckpt_name)

dataObj, _, _, _ = load_dataset(1,5)

#print(dataObj.train.features)
#print(dataObj.test.features)
#print(dataObj.validation.features)


sample = dataObj.test.features
label = dataObj.test.labels

tf.reset_default_graph()
estimated = nog.net_output_gen('reshape_y/y_conv:0',['x:0','dropout/keep_prob:0'],[sample, 0.5], ckpt_path, 'name')

#### y_conv analysis
#estimated = estimated[0, :, :, :]
#plt.figure(1)
#plt.imshow(estimated)

fig = plt.figure(1)
ax = plt.subplot(221)
ax.set_title('estimated1')
plt.imshow(estimated[0, :, :, 1])
ax = plt.subplot(222)
ax.set_title('label1')
plt.imshow(label[0, :, :, 1])
ax = plt.subplot(223)
ax.set_title('estimated2')
plt.imshow(estimated[0, :, :, 2])
ax = plt.subplot(224)
ax.set_title('label2')
plt.imshow(label[0, :, :, 2])

