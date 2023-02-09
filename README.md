# MitralValveSegmentation
Advanced Machine Learning Task 4

## Description

In this task, we had to segment the mitral valve in each video frame. This is challenging because we are given less than 200 frames with valve segmentations. However, we do have a lot more frames with corresponding box segmentations. We solve this task by training a DeepLabv3+ model with a resnet101 encoder to segment the boxes first. This allows the network to learn the rough location of the mitral valve.
Once the network converges, we save the weights and use them as our initial weights to train the same architecture again but this time on the valve segmentations. We do not differentiate between amateur and expert annotations (which are different quality data available in the dataset).
We notice from the results that the network performs very well and we get a 0.49 score but occasionally it segments far away pixels as mitral valves. To fix this, we use our box-segmenter network to first predict the box and then feed the image to the valve-segmenter network to predict the valves. We use the two output predictions and apply the AND operator to ensure that we only have segmentations within the box.
For thresholding the box segmenter’s sigmoid output (to obtain a binary mask), we use a threshold of 0.2. For the valve-segmenter output, we use a threshold of 0.3.

The videos in the dataset are of different sizes, so we resize them before and after feeding the images through the network.

## Result

Rank: 9/~130

Test video created by resizing all videos to a fixed size and applying the Boolean segmentation mask over each frame. The mitral valve predicted segmentation is thus seen in white above the left ventricle.

https://user-images.githubusercontent.com/30126243/217818931-872edaec-4284-4582-8beb-1ea15dad3a12.mp4

