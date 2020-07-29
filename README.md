# YOLOv3_training_code
This is the PyTorch implementation of YOLOv3, with support for training, inference and evaluation. The model in this repository has been trained on <a href = "https://motchallenge.net/data/MOT17Det/">MOT17Det</a> and <a href = "https://motchallenge.net/data/MOT20Det/">MOT20Det</a> datasets. Moreover, some parts of the code has been taken from <a href = "https://github.com/eriklindernoren/PyTorch-YOLOv3">this repository</a>.

## Weights File
The weights file obtained by training YOLOv3 using this repository can be obtained using the following link:<br>
https://drive.google.com/file/d/1-CB_Qz2W-zMUny4l9mjHEJKqp5jZgDS-/view?usp=sharing

## Darknet-53 Feature Extractor weights file
The model is trained by making use of transfer learning i.e., a feature extractor network pretrained on 1000 class Imagenet dataset has been used, the weights of which have been provided by the author's of YOLOv3 and can be downloaded using the following link:<br>
https://pjreddie.com/media/files/darknet53.conv.74

## Detections Visualization
Some of the results obtained using this model can be found on the following link:<br>
https://drive.google.com/file/d/1YhGsXUKkmVWvByxZYcsXWjxrjveso476/view?usp=sharing
