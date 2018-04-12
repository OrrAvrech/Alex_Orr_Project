# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:29:53 2018

@author: sorrav
"""

import tensorflow as tf

def define_summaries(loss, accuracy, labels, logits):
      tf.summary.scalar("loss", loss)
      tf.summary.scalar("accuracy", accuracy)
      tf.summary.image('label1', tf.cast(tf.expand_dims(labels[:,:,:,0], axis=-1)*4,tf.uint8))
      tf.summary.image('label2', tf.cast(tf.expand_dims(labels[:,:,:,1], axis=-1)*4,tf.uint8))
      tf.summary.image('logit1', tf.cast(tf.expand_dims(logits[:,:,:,0], axis=-1)*4,tf.uint8))
      tf.summary.image('logit2', tf.cast(tf.expand_dims(logits[:,:,:,1], axis=-1)*4,tf.uint8))