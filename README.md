# A Deep Convolutional Encoder-Decoder Network for Blind Source Separation in Fluorescence Microscopy Images
<p align="center">
  <img src="./NNflow/examples/pred_29.png">
</p>


## Introduction
In fluorescence microscopy, a great number of experiments require the ability to identify and localize single fluorescent particles at the nanometer scale.
However, images may contain highly overlapped emitters and a separation is essential for precise localization. 
This project focuses on the separation of point source images within the Tetrapod PSFs family [[1]](#references). Once successful separation is achieved, established 3D localization techniques can be employed, facilitating accurate localization of these overlapped PSFs.
The fluorescent particle (emitter) separation problem is treated in two stages. First, we tackle the problem by estimating the linear transformation given by independent component analysis (ICA). Second, we propose a deep convolutional neural network (CNN) approach and apply it to the source separation problem. 

## ICA Approach
Lidke et al. showed that the blinking of quantum dots can be analyzed by ICA for separation, identification, and precise localization of each individual particle [[2]](#references). However, for a rather complicated task of highly overlapped objects in each frame, ICA is likely to fail. Other than the strong assumptions of a linear transformation model and the sources being statistically independent, the two previously mentioned ambiguities cause an inherent problem- the independent components should be matched to their corresponding ground truth sources for the sake of the algorithm's performance evaluation. Nevertheless, the order of the independent components cannot be determined and a matching criterion is needed, yet hard to obtain, since the "correct" scaling cannot be recovered.

## Architecture
The model is implemented as an encoder-decoder network, comprised of a convolutional network part, which acts as a feature extractor and a deconvolutional network part, which produces a source separation of images from the extracted features. 
The input layer size is 64x64x64, for 64 frames in each video input sample, and the output layer is 64x64x32 since each sample was generated with a maximum of 32 sources.

## Results
<p align="center">
  <img width=250 height=600 src="./NNflow/examples/test_6_ica.png">
</p>
<p align="center">
    <em> (Left) ground truth sources, (middle) network predictions, (right) ICA estimation of the input. Both the location (x,y coordinates) and structure of the Tetrapod are precisely reconstructed by our network. </em>
</p>

## References
[[1]](https://pubs.acs.org/doi/10.1021/acs.nanolett.5b01396) *Y. Shechtman, L. E. Weiss, A. S. Backer, S. J. Sahl, and Moerner W. E. “Precise Three-Dimensional Scan-Free Multiple-Particle Tracking over Large Axial Ranges with Tetrapod Point Spread Functions"*

[[2]](https://pubmed.ncbi.nlm.nih.gov/19498727/) *K. A. Lidke, B. Rieger, T. M. Jovin, and R. Heintzmann. “Superresolution by localization of quantum dots using blinking statistics”*
