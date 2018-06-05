import tensorflow as tf
# Import Models
import Models_keras as models
from tensorflow.python.keras import backend as K

def fit_model(cfg):
    
      # Import data
      epochs = cfg.exp.epochs
      dataObj = cfg.data.obj
      imgSize = cfg.data.imgSize
      numFrames = cfg.data.numFrames
      maxSources = cfg.data.maxSources
      data_params = [imgSize, numFrames, maxSources]
      print("loaded data with the following params:")
      print("imgSize is:" +str(imgSize))
      print("numFrames is:" +str(numFrames))
      print("maxSources is:" +str(maxSources))
      batch_size = cfg.exp.batch
      
      # Create Model
      arch_func = models.DeconvN
      model = arch_func(data_params)
      
      # Use Keras to train the model.
      validation_data = (dataObj.validation.features, dataObj.validation.labels)
      fitness = model.fit(x=dataObj.train.features,
                        y=dataObj.train.labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=validation_data)
      
      # Get the accuracy on the validation-set
      # after the last training-epoch.
      accuracy = fitness.history['val_acc'][-1]
      
      # Delete the Keras model with these hyper-parameters from memory.
      del model
    
      # Clear the Keras session, otherwise it will keep adding new
      # models to the same TensorFlow graph each time we create a model
      K.clear_session()
    
      return accuracy