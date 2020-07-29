import os
import numpy as np
from skimage import io
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.use('TkAgg')

det_dataset1 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-02/img1/'
path1 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-02/labels/'

det_dataset2 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-04/img1/'
path2 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-04/labels/'

det_dataset3 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-05/img1/'
path3 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-05/labels/'

det_dataset4 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-09/img1/'
path4 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-09/labels/'

det_dataset5 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-10/img1/'
path5 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-10/labels/'

det_dataset6 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-11/img1/'
path6 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-11/labels/'

det_dataset7 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-13/img1/'
path7 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-13/labels/'

def visualize(det_dataset, path, width, height):

	all_images_names = sorted(os.listdir(det_dataset))
	all_ann_names = sorted(os.listdir(path))

	plt.ion() # Interative Mode On
	fig = plt.figure()
	ax = fig.add_subplot(111)

	for i in range(len(all_images_names)):
		
		orig_image = Image.open(det_dataset + all_images_names[i])
		orig_image_size = orig_image.size
		ax.imshow(orig_image)
		detections = open(path + all_ann_names[i], 'r').readlines()
		n_detections = len(detections)
		colors = np.random.rand(n_detections,3)

		for d_num, det in enumerate(detections):
			det = det.split(' ')
			x_c = float(det[1])
			y_c = float(det[2])
			w = float(det[3])
			h = float(det[4])
			x_min = (x_c - w/2) * width
			y_min = (y_c - h/2) * height
			w = w * width
			h = h * height

			ax.add_patch(patches.Rectangle((x_min, y_min), w, h, fill = False, lw = 2, ec = colors[d_num, :]))

		fig.canvas.flush_events()	
		plt.draw()
		ax.cla()

visualize(det_dataset2, path2, width = 1920, height = 1080)