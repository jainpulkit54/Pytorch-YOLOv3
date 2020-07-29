import os
import torch
import random
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def resize(image, size):

	img = torch.nn.functional.interpolate(image.unsqueeze(0), size = size, mode = "nearest").squeeze(0)
	return img

class ImageFolder(Dataset):
	
	def __init__(self, folder_path, img_size):
		
		self.folder_path = folder_path
		self.files = sorted(os.listdir(self.folder_path))
		self.img_size = img_size
		self.resize = torchvision.transforms.Resize(size = (self.img_size, self.img_size))
		self.totensor = torchvision.transforms.ToTensor()

	def __getitem__(self, index):
		
		img_name = self.files[index]
		img = Image.open(self.folder_path + img_name)
		img = self.resize(img)
		img = self.totensor(img)

		return img

	def __len__(self):
		
		return len(self.files)

class myDataset(Dataset):
	
	def __init__(self, train_images, img_size, multiscale, augmentation = True):

		self.train_images = train_images
		self.img_size = img_size
		self.min_size = self.img_size - 3 * 32
		self.max_size = self.img_size + 3 * 32
		self.batch_count = 0
		self.augmentation = augmentation
		if multiscale == "True":
			self.multiscale = True
		elif multiscale == "False":
			self.multiscale = False

		with open(self.train_images, "r") as file:
			self.img_files = file.readlines()

		self.label_files = []
		for path in self.img_files:
			self.label_files.append(path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt"))

		self.resize = torchvision.transforms.Resize((img_size, img_size))
		self.colorjitter = torchvision.transforms.ColorJitter(brightness = 1.5, saturation = 1.5, hue = 0.1)
		self.horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p = 1.0)
		self.totensor = torchvision.transforms.ToTensor()

	def __getitem__(self, index):

		img_path = self.img_files[index].rstrip()
		img = Image.open(img_path).convert('RGB')
		img = self.resize(img)
		label_path = self.label_files[index].rstrip()
		try:
			targets = None
			if os.path.exists(label_path):
				boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
				targets = torch.zeros(boxes.shape[0], 6)
				targets[:, 1:] = boxes

			# Applying Augmentation to the images and bounding boxes
			if self.augmentation:
				img = self.colorjitter(img)
				if np.random.random() >= 0.5:
					img = self.horizontal_flip(img)
					targets[:, 2] = 1 - targets[:, 2]
		except:
			pass

		img = self.totensor(img)

		return img, targets

	def collate_fn(self, batch):

		imgs, targets = list(zip(*batch))
		
		# Remove the empty targets i.e., images with no annotations in ground truth
		targets = [boxes for boxes in targets if len(boxes) > 0]
		
		# Adding the sample index to targets
		for index, boxes in enumerate(targets):
			boxes[:, 0] = index
		targets = torch.cat(targets, 0)
		
		# Selects new image size for multiscale training every tenth batch
		if self.multiscale:
			if self.batch_count > 1 and self.batch_count % 10 == 0:
				self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
		
		# Resize images to input shape
		imgs = torch.stack([resize(img, self.img_size) for img in imgs])
		self.batch_count += 1
		
		return imgs, targets

	def __len__(self):

		return len(self.img_files)