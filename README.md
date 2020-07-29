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

## Training the Model

<code>usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--network_config NETWORK_CONFIG]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--use_pretrained_weights USE_PRETRAINED_WEIGHTS]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--pretrained_weights PRETRAINED_WEIGHTS]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--use_pretrained_backbone USE_PRETRAINED_BACKBONE]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--pretrained_backbone PRETRAINED_BACKBONE] [--n_cpu N_CPU]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--inp_img_size INP_IMG_SIZE]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--multiscale_training MULTISCALE_TRAINING]</code><br>

For a detailed description of the arguments, kindly refer to the <code>train.py</code> file.

## Inference

## Evaluate

## Credits
