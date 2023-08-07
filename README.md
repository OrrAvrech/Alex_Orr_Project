# A Deep Convolutional Encoder-Decoder Network for Blind Source Separation in Fluorescence Microscopy Images



## Introduction
In fluorescence microscopy, a great amount of experiments require the ability to identify and localize single fluorescent particles at the nanometer scale.
However, images may contain highly overlapped emitters and a separation is essential for precise localization. 
In this project, we are interested in a separation of point source images from the Tetrapod PSFs family [1]. After precise separation is achieved, known 3D localization techniques may be applied. In this manner, an accurate localization of overlapped PSFs can be accomplished.
The fluorescent particle (emitter) separation problem is treated in two stages. First, we tackle the problem by estimating the linear transformation given by independent component analysis (ICA). Second, we propose a deep convolutional neural network (CNN) approach and apply it on the source separation problem. 

## Architecture
The model is implemented as an encoder-decoder network, comprised of a convolutional network part, which acts as a feature extractor and a deconvolutional network part, which produces a source separation of images from the extracted features. 
The input layer size is 64x64x64, for 64 frames in each video input sample, and the output layer is 64x64x32, since each sample was generated with a maximum of 32 sources.

## Results
