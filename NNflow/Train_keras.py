from tensorflow.python.keras import backend as K
# Import Models
import Models_keras as models

#%% Create and Fit Model
def fit_model(cfg):
    
      # Import data
      epochs = cfg.exp.epochs
      dataObj = cfg.data.obj
      batch_size = cfg.exp.batch      
          
      # Create Model
      arch_func = models.DeconvN
      model = arch_func(cfg)
      
      # Use Keras to train the model.
      validation_data = (dataObj.validation.features, dataObj.validation.labels)
      fitness = model.fit(x=dataObj.train.features,
                          y=dataObj.train.labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data)
                      
      # Get the accuracy on the validation-set
      # after the last training-epoch.
      accuracy = fitness.history['val_cross_corr'][-1]
      
      # TODO: do-if accuracy is better
      # Save the new model to harddisk.
      model.save(cfg.paths.best_model)
      
      # Delete the Keras model with these hyper-parameters from memory.
      del model
    
      # Clear the Keras session, otherwise it will keep adding new
      # models to the same TensorFlow graph each time we create a model
      K.clear_session()
    
      return accuracy