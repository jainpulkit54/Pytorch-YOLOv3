import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, default = 'input_000212/', help = 'Enter the path of the folder where you want to store images')
parser.add_argument('--video_name', type = str, default = '/home/pulkit/Datasets/Release_Public_Dataset/Test/000213/000213.avi', help = 'Enter the path of the video whose frames you want to extract and store in folder')
args = parser.parse_args()

os.makedirs(args.path, exist_ok = True)

cap = cv2.VideoCapture(args.video_name)
frame_num = 0

while(cap.isOpened()):
	
	frame_num += 1
	ret, frame = cap.read()
	if ret == False:
		break
	cv2.imwrite(args.path + str(frame_num).zfill(6) + '.jpg', frame)
	
cap.release()
cv2.destroyAllWindows()