# Training a deep-learning classifier for aerial top view detection of vehicles

**CODE TO FOLLOW...**

## Overview

Deep learning approaches have demonstrated state-of-the-art performance in various computer vision tasks such as object detection and recognition. In this repository I provide code and details on how to develop and train a Convolutional Neural Network (CNN) to detect top-view vehicles from UAV footage.

<img src="./images/cnn.png" width="512">


### Convolutional Neural Network


<img src="https://cdn-images-1.medium.com/max/800/1*awybeIxq_Yvg8jBfvrzPjg.png" width="512">

### Color thresholding

To identify potential regions of interest such as the road, which is more probable to containg vehicles color thresholding is performed. But first it is necessary to identify the color regions that represent the area we are looking for. For this reason the *sliders_color.py* implements a GUI which takes as input an image and uses slider bars for the minimum and maximum pixels values per 3 color channesl in order to identigy the range of colors to isolate. The specific chromatic model used is the [HSV model](https://en.wikipedia.org/wiki/HSL_and_HSV). 

<img src="./images/color.png" width="512">

## Dependencies

- Keras - 2.2.0
- Tensorflow - 1.5.0
- Numpy - 1.14.5
- OpenCV - 3.4.0

## Demo

A demo of a larger scale training and data set is shown in the following video:

<a href="https://youtu.be/x3_ujmXM8xk
" target="_blank"><img src="https://cdn-images-1.medium.com/max/800/1*5QjytkBi1bXXiyGm6fohJA.jpeg" 
alt="IMAGE ALT TEXT HERE" width="240" height="240" border="10" /></a>


## Relevant Material

If you use this dataset and/or code in your research please site the following paper:

• C. Kyrkou, S. Timotheou, P. Kolios, T. Theocharides and C. G. Panayiotou, "Optimized vision-directed deployment of UAVs for rapid traffic monitoring," 2018 IEEE International Conference on Consumer Electronics (ICCE), Las Vegas, NV, 2018, pp. 1-6.
doi: 10.1109/ICCE.2018.8326145

[See Paper](https://ieeexplore.ieee.org/abstract/document/8326145)

Also more technical details can be found in the following Medium post:

[Medium Article](https://medium.com/@ckyrkou/training-a-deep-learning-classifier-for-aerial-top-view-detection-of-vehicles-874f88d81c4)
