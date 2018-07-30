import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

#%% Accuracy Metrices
def NCC(y_pred, y_label):
    """Normalized Cross Correlation (tested)"""
    Npred  = (y_pred - K.mean(y_pred, [1,2], keepdims=True)) / K.std(y_pred, [1, 2], keepdims=True)
    Nlabel = (y_label - K.mean(y_label, [1,2], keepdims=True)) / K.std(y_label, [1, 2], keepdims=True)
    res = K.abs(K.mean(Npred * Nlabel, [1, 2], keepdims=True))
    return K.mean(res)

#%% Deconvolutional Network model
def DeconvN(cfg, num_conv_Bulks, _kernel_size, _activation):
  """DeconvN builds the graph for a deconvolutional net for seperating emitters.
  Args:
    data_params: [imgSize, numFrames, maxSources]  
    x: an input tensor of shape 64 x 64 x numFrames
  Returns:
    y_conv: a tensor of shape 64 x 64 x maxSources 
  """
  
  # Contsants (per dataset)
  maxSources = cfg.data.maxSources
  numFrames  = cfg.data.numFrames
  imgSize    = cfg.data.imgSize
  
  # Hyperparams  
  
  # Data Dimensions
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - it would be 1 for grayscale
  # 3 for an RGB image, 4 for RGBA, numFrames for a movie.   
  input_shape_full = (imgSize, imgSize, numFrames)
  
  # Start construction of a Keras Sequential model.
  model = tf.keras.Sequential()
  
  model.add(layers.InputLayer(input_shape=(input_shape_full)))
  
  # First convolutional layer.
  # TODO: Categorial activations
  for i in range(num_conv_Bulks):
      model.add(layers.Conv2D(kernel_size=_kernel_size, strides=1, filters=32, padding='same',
                         activation=_activation, name='layer_conv1_1_{0}'.format(i+1))) 
      model.add(layers.Conv2D(kernel_size=_kernel_size, strides=1, filters=32, padding='same',
                         activation=_activation, name='layer_conv1_2_{0}'.format(i+1)))
      model.add(layers.MaxPooling2D(pool_size=2, strides=2))
      # Maps to 32x32x32 feature map
      
      # Second convolutional layer.
      model.add(layers.Conv2D(kernel_size=_kernel_size, strides=1, filters=64, padding='same',
                         activation=_activation, name='layer_conv2_1_{0}'.format(i+1))) 
      model.add(layers.Conv2D(kernel_size=_kernel_size, strides=1, filters=64, padding='same',
                         activation=_activation, name='layer_conv2_2_{0}'.format(i+1)))
      model.add(layers.MaxPooling2D(pool_size=2, strides=2))
      # Maps to 16x16x64 feature map
      
      # Third convolutional layer.
      model.add(layers.Conv2D(kernel_size=_kernel_size, strides=1, filters=128, padding='same',
                         activation=_activation, name='layer_conv3_1_{0}'.format(i+1))) 
      model.add(layers.Conv2D(kernel_size=_kernel_size, strides=1, filters=128, padding='same',
                         activation=_activation, name='layer_conv3_2_{0}'.format(i+1)))
      model.add(layers.MaxPooling2D(pool_size=2, strides=2))
  # Maps to 8x8x128 feature map
  
  # First Deconvolutional Layer + Unpsampling
  model.add(layers.UpSampling2D((2, 2)))
  model.add(layers.Conv2DTranspose(kernel_size=_kernel_size, strides=1, filters=128, padding='same',
                     activation=_activation, name='layer_deconv1_1')) 
  model.add(layers.Conv2DTranspose(kernel_size=_kernel_size, strides=1, filters=64, padding='same',
                     activation=_activation, name='layer_deconv1_2')) 
  
  # Second Deconvolutional Layer + Unpsampling
  model.add(layers.UpSampling2D((2, 2)))
  model.add(layers.Conv2DTranspose(kernel_size=_kernel_size, strides=1, filters=64, padding='same',
                     activation=_activation, name='layer_deconv2_1')) 
  model.add(layers.Conv2DTranspose(kernel_size=_kernel_size, strides=1, filters=32, padding='same',
                     activation=_activation, name='layer_deconv2_2')) 
  
  # First Deconvolutional Layer + Unpsampling
  #TODO: does this layer should get kernel size from outside?
  model.add(layers.UpSampling2D((2, 2)))
  model.add(layers.Conv2DTranspose(kernel_size=_kernel_size, strides=1, filters=32, padding='same',
                     activation=_activation, name='layer_deconv3_1')) 
  model.add(layers.Conv2DTranspose(kernel_size=_kernel_size, strides=1, filters=32, padding='same',
                     activation=_activation, name='layer_deconv3_2')) 
  model.add(layers.Conv2DTranspose(kernel_size=_kernel_size, strides=1, filters=maxSources, padding='same',
                     activation='linear', name='layer_deconv3_3')) 
  
  # Use the Adam method for training the network.
  # We want to find the best learning-rate for the Adam method.
  # TODO: learning rate param
  optimizer = Adam(lr=1e-3)
  
  # In Keras we need to compile the model so it can be trained.
  model.compile(optimizer=optimizer,
                  loss='mean_absolute_error',
                  metrics=[NCC])
  
  return model
