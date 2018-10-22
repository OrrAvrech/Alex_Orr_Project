# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 01:05:30 2018

@author: orrav
"""

#%%
import numpy as np
import h5py
import os
import pickle

def datasetFromMat(path, start_idx, end_idx, test_perc):
        
    dataset_x = []
    dataset_y = []
    
    for i in range(start_idx, end_idx):
        if i % np.floor((end_idx-start_idx)/10) == 0: 
            print ("loaded %d/%d samples" % (i-start_idx+1, (end_idx-start_idx)))
            
        data_file = os.path.join(path, str(i)+ '.mat')
        with h5py.File(data_file, 'r') as f:
            sample_x = np.array(f.get('features'))
            sample_y = np.array(f.get('labels'))
            
            sample_y = np.reshape(sample_y, (1,sample_y.shape[0],sample_y.shape[1],sample_y.shape[2]))            
            sample_y = np.transpose(sample_y, (0, 2, 3, 1))
            # sample_y : shape[2] is maxSources
            
            sample_x = np.reshape(sample_x, (1,sample_x.shape[0],sample_x.shape[1],sample_x.shape[2]))
            sample_x = np.transpose(sample_x, (0, 2, 3, 1))
            # sample_y : shape[2] is numFrames            
            
            if i == start_idx:
                dataset_x = sample_x
                dataset_y = sample_y
            else:
                dataset_x = np.append(dataset_x,sample_x, axis=0)
                dataset_y = np.append(dataset_y,sample_y, axis=0)
    # Normalized features
    features = (dataset_x - np.mean(dataset_x))/np.std(dataset_x)
    labels = dataset_y
    imgSize = features.shape[1]
    numFrames = features.shape[3]
    maxSources = labels.shape[3]
    dataset_size = features.shape[0]
    test_size = np.rint(test_perc * dataset_size).astype(int) # round to int
    train_size = dataset_size - test_size  
                                        
    train_features, test_features = np.split(features, [train_size]) 
    train_labels, test_labels = np.split(labels, [train_size])       
                                               
    datasetMat = type('', (), {})()
    datasetMat.train_features = train_features
    datasetMat.train_labels   = train_labels
    datasetMat.test_features  = test_features
    datasetMat.test_labels    = test_labels
    return datasetMat, imgSize, numFrames, maxSources

class DataSet(object):

  def __init__(self,
               features,
               labels):
    """Construct a DataSet"""
    assert features.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (features.shape, labels.shape))
    self._num_examples = int(features.shape[0])
#    print ("_num examples is : "+str(features.shape[0]))
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._features = self.features[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      features_rest_part = self._features[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._features = self.features[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      features_new_part = self._features[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((features_rest_part, features_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._features[start:end], self._labels[start:end]


def read_data_sets(path, start_idx, end_idx, loadObj, saveObj2file):

  # Parameters: validation set and test set sizes  
  test_perc = 0.2
  validation_perc = 0.2 # out of train size
  
  if loadObj == True:
      with open(os.path.join(path, 'train' + '.obj'), 'rb') as trainFile:
        train = pickle.load(trainFile)
        imgSize = train.labels.shape[1]
        maxSources = train.labels.shape[-1]
        numFrames = train.features.shape[-1]
      with open(os.path.join(path, 'validation' + '.obj'), 'rb') as validationFile:
        validation = pickle.load(validationFile)
      with open(os.path.join(path, 'test' + '.obj'), 'rb') as testFile:
        test = pickle.load(testFile)
  else:
      datasetMat, imgSize, numFrames, maxSources = datasetFromMat(path, start_idx, end_idx, test_perc)  
    
      train_features = datasetMat.train_features
      train_labels   = datasetMat.train_labels  
      test_features  = datasetMat.test_features
      test_labels    = datasetMat.test_labels
        
      validation_size = np.rint(validation_perc * np.float_(train_features.shape[0])).astype(int)
      
      validation_features = train_features[:validation_size]
      validation_labels = train_labels[:validation_size]
      train_features = train_features[validation_size:]
      train_labels = train_labels[validation_size:]
    
      train = DataSet(train_features, train_labels)
      validation = DataSet(validation_features, validation_labels)
      test = DataSet(test_features, test_labels)
      
      # Saving dataObj to a file (for reusing it later on)
      if saveObj2file == True:
          trainFile = open(os.path.join(path, 'train.obj'), 'wb') # in Windows use 'w'/'r' only for text files
          pickle.dump(train, trainFile, protocol=4)
          validationFile = open(os.path.join(path, 'validation.obj'), 'wb')
          pickle.dump(validation, validationFile, protocol=4)
          testFile = open(os.path.join(path, 'test.obj'), 'wb')
          pickle.dump(test, testFile, protocol=4)
    
  dataObj = type('', (), {})()
  dataObj.train = train
  dataObj.validation = validation
  dataObj.test = test
  
  return dataObj, imgSize, numFrames, maxSources

def load_dataset(start_idx, num_samp, path, loadObj, saveObj2file=False):
  end_idx = start_idx + num_samp
  return read_data_sets(path, start_idx, end_idx, loadObj, saveObj2file)

#%% ICA Load
from scipy.io import loadmat

ICA_File = os.path.join('search_results', 'IC.mat')
est = loadmat(ICA_File)['IC']
labelMatchList = [3, 24, 29, 20, 18, 5, 23, 17, 28, 3, 22, 11, 25, 15, 9, 7, 10, 12, 1, 8, 6, 14, 27, 30, 
                  31, 13, 4, 26, 2, 32, 21, 19]
labelMatchList[:] = [x - 1 for x in labelMatchList]
estSorted = np.zeros_like(est)
estSorted = est[:,:,labelMatchList]

ICA_sorted_File = open(os.path.join('search_results', 'ICA_est.obj'), 'wb')
pickle.dump(estSorted, ICA_sorted_File, protocol=4)