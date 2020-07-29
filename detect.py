import os
import time
import torch
import argparse
import matplotlib
import numpy as np
import matplotlib.patches as patches
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import *
from networks import *
from utils import *
from PIL import Image, ImageDraw
#matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 1, help = "Image Batch Size")
parser.add_argument("--image_folder", type = str, default = "/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/test/MOT17-14/img1/", help = "Path to the dataset folder")
parser.add_argument("--text_file_path", type = str, default = "data/detections/", help = "This is the text file where the detections will be stored in MOT format")
parser.add_argument("--network_config", type = str, default = "config/yolov3-custom.cfg", help = "Patch to the file containing the network definition")
parser.add_argument("--weights_path", type = str, default = "checkpoints_800_mot1720Det/model_epoch_29.pth", help = "Path to the weights file")
parser.add_argument("--class_path", type = str, default = "data/classes.names", help = "Path to the class label file")
parser.add_argument("--conf_thresh", type = float, default = 0.5, help = "Object Confidence Threshold")
parser.add_argument("--nms_thresh", type = float, default = 0.5, help = "IOU threshold for Non-Maximum Suppression")
parser.add_argument("--n_cpu", type = int, default = 0, help = "Number of CPU threads to use for batch generation")
parser.add_argument("--inp_img_size", type = int, default = 800, help = "Dimension of input image to the network")
parser.add_argument("--display", type = str, default = 'False', help = "Set this to True if want to visualize detections on image")
args = parser.parse_args()

if args.display == 'False':
	args.display = False
else:
	args.display = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('output_000213', exist_ok = True)
os.makedirs(args.text_file_path, exist_ok = True)

# Model Initialization
model_config = args.network_config
img_size = args.inp_img_size
model = Darknet(model_config, img_size)

# Loading the checkpoint weights
checkpoint = torch.load(args.weights_path)
model_parameters = checkpoint['model_state_dict']
model.load_state_dict(model_parameters)
model.to(device)
model.eval()

images = ImageFolder(args.image_folder, img_size)
dataloader = DataLoader(images, batch_size = 1, shuffle = False, num_workers = args.n_cpu)

with open(args.class_path, 'r') as class_name_file:
	names = class_name_file.readlines()

class_names = []
for name in names:
	class_names.append(name.rstrip().lstrip())

images_names = sorted(os.listdir(args.image_folder))

# This is the text file to store the detections in MOT format to be used by the SORT tracker
text_file_path = args.text_file_path
file_det = open(text_file_path + 'detections.txt', 'w+')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
print('Performing Object Detection\n')

#plt.ion() # Interative Mode On
fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)

for index, image in enumerate(dataloader):
	
	img = image.type(Tensor)
	orig_image = Image.open(args.image_folder + images_names[index])
	orig_image_size = orig_image.size

	ax.imshow(orig_image)
	ax.set_axis_off()
	#plt.title('Detected Objects in Frame ' + str(index+1))
	
	tic = time.time()
	
	with torch.no_grad():
		detections = model(img)
		detections = non_max_suppression(detections, args.conf_thresh, args.nms_thresh)

	toc = time.time()
	print('Processing Time',toc - tic)

	try:
		detections = detections[0].numpy()
		# The detections contain 7 columns which are [x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num]
		# and the bounding box coordinates are in 'opt.img_size' scale 
		unique_classes = np.unique(detections[:,-1])
		n_detections = detections.shape[0]
		colors = np.random.rand(n_detections,3)

		for d_num, (x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num) in enumerate(detections):
			
			cls_num = int(cls_num)
			
			if cls_num == 0:
				name = class_names[0]
			elif cls_num == 1:
				name = class_names[1]

			if cls_num == 0:
				x_min, y_min, x_max, y_max = x_min/args.inp_img_size, y_min/args.inp_img_size, x_max/args.inp_img_size, y_max/args.inp_img_size
				x_min = x_min * orig_image_size[0]
				y_min = y_min * orig_image_size[1]
				x_max = x_max * orig_image_size[0]
				y_max = y_max * orig_image_size[1]
				width = x_max - x_min
				height = y_max - y_min
			
				ax.add_patch(patches.Rectangle((x_min, y_min), width, height, fill = False, lw = 3, ec = colors[d_num, :]))
				#plt.text(x = x_min, y = y_min, s = name + str('%0.2f'%cls_conf))
			
				file_det.write(str(index+1)+','+str(cls_num)+','+str('%0.2f'%x_min)+','+str('%0.2f'%y_min)+','+str('%0.2f'%width)+','+str('%0.2f'%height)+','+str('%0.2f'%cls_conf)+','+'-1'+','+'-1'+','+'-1')
				file_det.write('\n')

		fig.savefig('output_000213/' + str(index + 1).zfill(6) + '.jpg', bbox_inches = 'tight', pad_inches = 0)

	except:
		pass

	#plt.draw()
	ax.cla()
	
	#if args.display:
	#	fig.canvas.flush_events()	