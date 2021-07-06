This is the PyTorch implementation of YOLOv3, with support for training, inference and evaluation. Some parts of the code has been taken from <a href = "https://github.com/eriklindernoren/PyTorch-YOLOv3">this repository</a>.

## Darknet-53 Feature Extractor weights file
The model is trained by making use of transfer learning i.e., a feature extractor network pretrained on 1000 class Imagenet dataset has been used, the weights of which have been provided by the author's of YOLOv3 and can be downloaded using the following link:<br>
https://pjreddie.com/media/files/darknet53.conv.74

## Dataset Preparation
Run the commands below to create a custom model definition, replacing <i>num_classes</i> with the number of classes in your dataset.<br>
<code>$ cd config</code><br>
<code>$ bash create_custom_model.sh num_classes</code><br>
### Classes
Add class names to <code>data/classes.names</code>. This file should have one class name per row.
### Image Folder
Move the images of your dataset to <code>data/images/</code> folder.
### Annotation Folder
Move the annotations of your dataset to <code>data/labels/</code> folder. The dataloader in this repository expects that the annotation file corresponding to the image <code>data/images/train.jpg</code> has the path <code>data/labels/train.txt</code>. Each row in the annotation file should define one bounding box, using the syntax <code>label_id x_center y_center width height</code>. The coordinates should be scaled to <code>[0, 1]</code>, and the label_id should be zero-indexed and correspond to the row number of the class name in <code>data/classes.names</code>
### Define Train and Validation Sets
In <code>data/train.txt</code> and <code>data/valid.txt</code>, add paths to images that will be used as train and validation data respectively.
## Train
<code>usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--network_config NETWORK_CONFIG]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--use_pretrained_weights USE_PRETRAINED_WEIGHTS]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--pretrained_weights PRETRAINED_WEIGHTS]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--use_pretrained_backbone USE_PRETRAINED_BACKBONE]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--pretrained_backbone PRETRAINED_BACKBONE] [--n_cpu N_CPU]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--inp_img_size INP_IMG_SIZE]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--multiscale_training MULTISCALE_TRAINING]</code><br>

For a detailed description of the arguments, kindly refer to the <code>train.py</code> file.

### Tensorboard
In order to track training progress in tensorboard:<br>
1) Initiate Training and switch to the directory where code is present.<br>
2) Run the command:<br>
<code>tensorboard --logdir logs</code><br>
3) Go the browser and type:<br>
<code>http://localhost:6006/</code>

## Inference
<code>usage: detect.py [-h] [--batch_size BATCH_SIZE] [--image_folder IMAGE_FOLDER]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--text_file_path TEXT_FILE_PATH]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--network_config NETWORK_CONFIG]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--n_cpu N_CPU] [--inp_img_size INP_IMG_SIZE]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--display DISPLAY]</code><br>

For a detailed description of the arguments, kindly refer to the <code>detect.py</code> file.

## Evaluate
<code>usage: test.py [-h] [--batch_size BATCH_SIZE]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--network_config NETWORK_CONFIG] [--weights_path WEIGHTS_PATH]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--iou_thresh IOU_THRESH] [--conf_thresh CONF_THRESH]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--nms_thresh NMS_THRESH] [--n_cpu N_CPU]</code><br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <code>[--inp_img_size INP_IMG_SIZE]</code><br>

For a detailed description of the arguments, kindly refer to the <code>test.py</code> file.

## Other Files
The files present in the folder <code>data_preparation/</code> are used for dataset preparation (in particular, MOT17Det and MOT20Det).<br>
The file <code>make_video.py</code> is used to make a video out of the detections obtained.

## Credits
## YOLOv3: An Incremental Improvement
<i>Joseph Redmon, Ali Farhadi</i>
### Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at<br>
https://pjreddie.com/darknet/yolo/<br>

<a href = "https://pjreddie.com/media/files/papers/YOLOv3.pdf">[Paper]</a>
<a href = "https://pjreddie.com/darknet/yolo/">[Project Webpage]</a>
<a href = "https://github.com/pjreddie/darknet">[Author's Implementation]</a><br>
<pre>
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
</pre>
