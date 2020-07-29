import os
import numpy as np

np.random.seed(42)
images_folder = '../data/images/'
images_names = sorted(os.listdir(images_folder))
indices = [i for i in range(len(images_names))]
np.random.shuffle(indices)

train_file = open('../data/train.txt', 'w+')
valid_file = open('../data/valid.txt', 'w+')

num_train_samples = int(0.8 * len(images_names))
num_valid_samples = len(images_names) - num_train_samples
train_indices = indices[0:num_train_samples]
valid_indices = indices[num_train_samples:]

for index in train_indices:
	if index == train_indices[-1]:
		train_file.write('data/images/' + images_names[index])
	else:
		train_file.write('data/images/' + images_names[index] + '\n')

for index in valid_indices:
	if index == valid_indices[-1]:
		valid_file.write('data/images/' + images_names[index])
	else:
		valid_file.write('data/images/' + images_names[index] + '\n')


