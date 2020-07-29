import os
import cv2
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

det_dataset1 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-01/img1/' # 25 fps
det_dataset1_gt = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-01/gt/gt.txt'
path1 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-01/labels/'
images1 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-01/images/'
os.makedirs(path1, exist_ok = True)
os.makedirs(images1, exist_ok = True)
images_names1 = sorted(glob.glob(det_dataset1 + '*.jpg'))
height1 = 1080
width1 = 1920

det_dataset2 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-02/img1/' # 25 fps
det_dataset2_gt = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-02/gt/gt.txt'
path2 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-02/labels/'
images2 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-02/images/'
os.makedirs(path2, exist_ok = True)
os.makedirs(images2, exist_ok = True)
images_names2 = sorted(glob.glob(det_dataset2 + '*.jpg'))
height2 = 1080
width2 = 1920

det_dataset3 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-03/img1/' # 14 fps
det_dataset3_gt = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-03/gt/gt.txt'
path3 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-03/labels/'
images3 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-03/images/'
os.makedirs(path3, exist_ok = True)
os.makedirs(images3, exist_ok = True)
images_names3 = sorted(glob.glob(det_dataset3 + '*.jpg'))
height3 = 880
width3 = 1173

det_dataset4 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-05/img1/' # 30 fps
det_dataset4_gt = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-05/gt/gt.txt'
path4 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-05/labels/'
images4 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-05/images/'
os.makedirs(path4, exist_ok = True)
os.makedirs(images4, exist_ok = True)
images_names4 = sorted(glob.glob(det_dataset4 + '*.jpg'))
height4 = 1080
width4 = 1654

def parse_gt_file(file_path):

	with open(file_path, 'r') as file:
		rows = file.readlines()

	sorted_rows = []
	
	for row in rows:
		row = row.rstrip().lstrip()
		row_list = row.split(',')
		r = []
		for element in row_list:
			r.append(float(element.rstrip().lstrip()))
		sorted_rows.append(r)

	sorted_rows = sorted(sorted_rows)
	sorted_rows = np.array(sorted_rows)
	return sorted_rows

def create_image_file(images_names, new_path, frame_count = 0):

	for i, name in enumerate(images_names):
		i = i + 1
		img = cv2.imread(name)
		cv2.imwrite((new_path + str(int(i + frame_count)).zfill(6) + '.jpg'), img)

def create_ann_file(r, path, width, height, frame_count = 0):

	frame_nos = np.unique(r[:,0])

	for frame_number in frame_nos:
		
		frame_count = frame_count + 1
		indices = np.where(r[:,0] == frame_number)[0]
		ann_file_name = path + str(int(frame_count)).zfill(6) + '.txt'
		ann_file = open(ann_file_name, 'w+')
		subarray = r[np.ix_(indices, [2,3,4,5,7])]
		
		for i in range(subarray.shape[0]):
			row = subarray[i,:]
			box_xmin = row[0]; box_ymin = row[1]; box_width = row[2]; box_height = row[3]; cls = row[4]
			box_xmin = np.clip(box_xmin, 0, width - 1)
			box_ymin = np.clip(box_ymin, 0, height - 1)
			box_xmax = box_xmin + box_width
			box_ymax = box_ymin + box_height

			if box_xmax >= width or box_ymax >= height:
				box_xmax = np.clip(box_xmax, 0, width - 1)
				box_ymax = np.clip(box_ymax, 0, height - 1)
				box_width = box_xmax - box_xmin
				box_height = box_ymax - box_ymin

			box_xmin = box_xmin/width
			box_ymin = box_ymin/height
			box_width = box_width/width
			box_height = box_height/height
			box_xmin = box_xmin + box_width/2
			box_ymin = box_ymin + box_height/2

			if cls == 1 or cls == 2 or cls == 7:
				if i == subarray.shape[0] - 1:
					towrite = str(0) + ' ' + str(box_xmin) + ' ' + str(box_ymin) + ' ' + str(box_width) + ' ' + str(box_height)
				else:
					towrite = str(0) + ' ' + str(box_xmin) + ' ' + str(box_ymin) + ' ' + str(box_width) + ' ' + str(box_height) + '\n'
				ann_file.write(towrite)
			elif cls == 3 or cls == 4 or cls == 5 or cls == 6 or cls == 8 or cls == 12:
				if i == subarray.shape[0] - 1:
					towrite = str(1) + ' ' + str(box_xmin) + ' ' + str(box_ymin) + ' ' + str(box_width) + ' ' + str(box_height)
				else:
					towrite = str(1) + ' ' + str(box_xmin) + ' ' + str(box_ymin) + ' ' + str(box_width) + ' ' + str(box_height) + '\n'
				ann_file.write(towrite)

	return frame_count


r1 = parse_gt_file(det_dataset1_gt)
create_image_file(images_names1, images1, frame_count = 0)
frame_count = create_ann_file(r1, path1, width1, height1, frame_count = 0)

r2 = parse_gt_file(det_dataset2_gt)
create_image_file(images_names2, images2, frame_count = frame_count)
frame_count = create_ann_file(r2, path2, width2, height2, frame_count = frame_count)

r3 = parse_gt_file(det_dataset3_gt)
create_image_file(images_names3, images3, frame_count = frame_count)
frame_count = create_ann_file(r3, path3, width3, height3, frame_count = frame_count)

r4 = parse_gt_file(det_dataset4_gt)
create_image_file(images_names4, images4, frame_count = frame_count)
frame_count = create_ann_file(r4, path4, width4, height4, frame_count = frame_count)