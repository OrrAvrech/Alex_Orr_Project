# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:29:53 2018

@author: sorrav
"""

import tensorflow as tf

import matplotlib
import matplotlib.cm

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value

def define_summaries(loss, accuracy, labels, logits):
      tf.summary.scalar("loss", loss)
      tf.summary.scalar("accuracy", accuracy)
      maxSources = labels.shape[-1]
      for i in range(maxSources - 1):
          label_color = colorize(labels[:,:,:,i], cmap='viridis')
          logit_color = colorize(logits[:,:,:,i], cmap='viridis')
          tf.summary.image('label' + str(i), tf.expand_dims(label_color, axis=0))
          tf.summary.image('logit' + str(i), tf.expand_dims(logit_color, axis=0))
      