import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_image_path', type = str, default = 'output_000213/', help = 'The folder path where images with detections are stored')
parser.add_argument('--output_video_name', type = str, default = 'output_000213.avi', help = 'The name of the video which you want to create')
parser.add_argument('--fps', type = str, default = '25', help = 'Enter the Frame Per Second')
args = parser.parse_args()

image_path = args.output_image_path
images_names = sorted(os.listdir(image_path))
video_name = args.output_video_name
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = args.fps
fps = float(fps)
video = cv2.VideoWriter(video_name, fourcc, fps, (1232, 693))

img_array = []

for img_name in images_names:
	
	frame = cv2.imread(image_path + img_name)
	video.write(frame)

video.release()
cv2.destroyAllWindows()