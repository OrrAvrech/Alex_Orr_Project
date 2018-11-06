import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

#%% Accuracy Metrices
def NCC(y_pred, y_label):
    """Normalized Cross Correlation (tested)"""
    Npred  = (y_pred - K.mean(y_pred, [1,2], keepdims=True)) #/ K.std(y_pred, [1, 2], keepdims=True)
    Nlabel = (y_label - K.mean(y_label, [1,2], keepdims=True))# / K.std(y_label, [1, 2], keepdims=True)
    res = K.abs(K.mean(Npred * Nlabel, [1, 2], keepdims=True))
    return K.mean(res)

#%% Deconvolutional Network model
def DeconvN(data_params, learning_rate, num_conv_Bulks, kernel_size, activation):
  """DeconvN builds the graph for a deconvolutional net for seperating emitters.
  Args:
    data_params: [imgSize, numFrames, maxSources]  
    x: an input tensor of shape 64 x 64 x numFrames
  Returns:
    y_conv: a tensor of shape 64 x 64 x maxSources 
  """
  
  # Contsants (per dataset)
  maxSources = data_params[2]
  numFrames  = data_params[1]
  imgSize    = data_params[0]
  
  # Hyperparams
  
  
  # Data Dimensions
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - it would be 1 for grayscale
  # 3 for an RGB image, 4 for RGBA, numFrames for a movie.   
  input_shape_full = (imgSize, imgSize, numFrames)
  
  # Start construction of a Keras Sequential model.
  model = tf.keras.Sequential()
  
  model.add(layers.InputLayer(input_shape=(input_shape_full)))
  
  for i in range(num_conv_Bulks):
      # Convolutional Layers + MaxPooling
      model.add(layers.Conv2D(kernel_size=kernel_size, strides=1, filters=32 * 2^i, padding='same',
                         activation=activation, name='layer_conv{0}_1'.format(i+1))) 
      model.add(layers.Conv2D(kernel_size=kernel_size, strides=1, filters=32 * 2^i, padding='same',
                         activation=activation, name='layer_conv{0}_2'.format(i+1)))
      model.add(layers.MaxPooling2D(pool_size=2, strides=2))
      
  for j in range(num_conv_Bulks-1,-1,-1):
      # Upsampling + Deconvolutional Layers
      model.add(layers.UpSampling2D((2, 2)))
      model.add(layers.Conv2DTranspose(kernel_size=kernel_size, strides=1, filters=32 * 2^j, padding='same',
                         activation=activation, name='layer_deconv{0}_1'.format(num_conv_Bulks-j))) 
      model.add(layers.Conv2DTranspose(kernel_size=kernel_size, strides=1, filters=32 * 2^j, padding='same',
                         activation=activation, name='layer_deconv{0}_2'.format(num_conv_Bulks-j))) 

  model.add(layers.Conv2DTranspose(kernel_size=kernel_size, strides=1, filters=maxSources, padding='same',
                     activation=activation, name='layer_deconv_output')) 
  
  # Use the Adam method for training the network.
  # We want to find the best learning-rate for the Adam method.
  optimizer = Adam(lr=learning_rate)
  
  # In Keras we need to compile the model so it can be trained.
  model.compile(optimizer=optimizer,
                  loss='mean_absolute_error',
                  metrics=['mae'])
  
  return model
