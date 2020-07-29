import os
import time
import torch
import argparse
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from networks import *
from utils import *
from datasets import *
from test import evaluate

writer = SummaryWriter('logs/YOLOv3')

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 30, help = "The number of Epochs")
parser.add_argument("--batch_size", type = int, default = 4, help = "The Image Batch Size")
parser.add_argument("--network_config", type = str, default = "config/yolov3-custom.cfg", help = "Patch to the file containing the network definition")
parser.add_argument("--use_pretrained_weights", type = str, default = "False", help = "If True initializes model with pretrained weights file")
parser.add_argument("--pretrained_weights", type = str, default = "checkpoints/weights.pth", help = "Path to the pretrained model weights")
parser.add_argument("--use_pretrained_backbone", type = str, default = "True", help = "If True initializes the backbone with pretrained Darknet53 weights")
parser.add_argument("--pretrained_backbone", type = str, default = "weights/darknet53.conv.74", help = "Path to the pretrained backbone weights")
parser.add_argument("--n_cpu", type = int, default = 8, help = "Number of CPU threads to use for Batch Generation")
parser.add_argument("--inp_img_size", type = int, default = 800, help = "Dimension of input image to the network")
parser.add_argument("--multiscale_training", type = str, default = "False", help = "Allow for Multi-Scale Training")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok = True)

# Get data configuration
train_images_name = 'data/train.txt'
valid_images_name = 'data/valid.txt'

with open('data/classes.names') as class_name_file:
	names = class_name_file.readlines()

class_names = []
for name in names:
	class_names.append(name.rstrip().lstrip())

# Model Initialization
model_config = args.network_config
img_size = args.inp_img_size
model = Darknet(model_config, img_size)

if args.use_pretrained_weights == "True":
	model.load_state_dict(torch.load(args.pretrained_weights))
else:
	if args.use_pretrained_backbone == "True":
		model.load_darknet_weights(args.pretrained_backbone)

train_dataset = myDataset(train_images_name, img_size, multiscale = args.multiscale_training, augmentation = True)
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, collate_fn = train_dataset.collate_fn)
val_dataset = myDataset(valid_images_name, img_size, multiscale = "False", augmentation = False)
val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, collate_fn = val_dataset.collate_fn)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)

def run_epoch(data_loader, model, optimizer, epoch_count = 0):
	
	model.to(device)
	model.train()

	for batch_id, (imgs, targets) in enumerate(data_loader):
		
		iter_count = epoch_count * len(data_loader) + batch_id
		imgs = imgs.to(device)
		targets = targets.to(device)
		loss, outputs = model(imgs, targets)
		optimizer.zero_grad()

		loss.backward()
		optimizer.step()

		# Code for Logging the Losses
		
		layers = model.yolo_layers
		print('------------Epoch', str(epoch_count), 'Batch', str(batch_id + 1), '/', len(data_loader), '---------------')
		table_data = [
		['Metric', 'YOLO-Layer-1', 'YOLO-Layer-2', 'YOLO-Layer-3'],
		['Grid Size', '%2d'%layers[0].metrics['grid_size'], '%2d'%layers[1].metrics['grid_size'], '%2d'%layers[2].metrics['grid_size']],
		['Overall Loss', '%0.5f'%layers[0].metrics['loss'], '%0.5f'%layers[1].metrics['loss'], '%0.5f'%layers[2].metrics['loss']],
		['x_loss', '%0.5f'%layers[0].metrics['x'], '%0.5f'%layers[1].metrics['x'], '%0.5f'%layers[2].metrics['x']],
		['y_loss', '%0.5f'%layers[0].metrics['y'], '%0.5f'%layers[1].metrics['y'], '%0.5f'%layers[2].metrics['y']],
		['w_loss', '%0.5f'%layers[0].metrics['w'], '%0.5f'%layers[1].metrics['w'], '%0.5f'%layers[2].metrics['w']],
		['h_loss', '%0.5f'%layers[0].metrics['h'], '%0.5f'%layers[1].metrics['h'], '%0.5f'%layers[2].metrics['h']],
		['confidence_loss', '%0.5f'%layers[0].metrics['conf'], '%0.5f'%layers[1].metrics['conf'], '%0.5f'%layers[2].metrics['conf']],
		['classification_loss', '%0.5f'%layers[0].metrics['cls'], '%0.5f'%layers[1].metrics['cls'], '%0.5f'%layers[2].metrics['cls']],
		['conf_obj', '%0.5f'%layers[0].metrics['conf_obj'], '%0.5f'%layers[1].metrics['conf_obj'], '%0.5f'%layers[2].metrics['conf_obj']],
		['conf_noobj', '%0.5f'%layers[0].metrics['conf_noobj'], '%0.5f'%layers[1].metrics['conf_noobj'], '%0.5f'%layers[2].metrics['conf_noobj']]
		]
		table = AsciiTable(table_data)
		print(table.table)
		print('Total Loss:', '%0.5f'%loss.item())

		# Logs for "YOLO-Layer-1"
		writer.add_scalar('YOLO_Layer_1_Overall_Loss', layers[0].metrics['loss'], iter_count)
		writer.add_scalar('YOLO_Layer_1_x_loss', layers[0].metrics['x'], iter_count)
		writer.add_scalar('YOLO_Layer_1_y_loss', layers[0].metrics['y'], iter_count)
		writer.add_scalar('YOLO_Layer_1_w_loss', layers[0].metrics['w'], iter_count)
		writer.add_scalar('YOLO_Layer_1_h_loss', layers[0].metrics['h'], iter_count)
		writer.add_scalar('YOLO_Layer_1_confidence_loss', layers[0].metrics['conf'], iter_count)
		writer.add_scalar('YOLO_Layer_1_classification_loss', layers[0].metrics['cls'], iter_count)
		# Logs for "YOLO-Layer-2"
		writer.add_scalar('YOLO_Layer_2_Overall_Loss', layers[1].metrics['loss'], iter_count)
		writer.add_scalar('YOLO_Layer_2_x_loss', layers[1].metrics['x'], iter_count)
		writer.add_scalar('YOLO_Layer_2_y_loss', layers[1].metrics['y'], iter_count)
		writer.add_scalar('YOLO_Layer_2_w_loss', layers[1].metrics['w'], iter_count)
		writer.add_scalar('YOLO_Layer_2_h_loss', layers[1].metrics['h'], iter_count)
		writer.add_scalar('YOLO_Layer_2_confidence_loss', layers[1].metrics['conf'], iter_count)
		writer.add_scalar('YOLO_Layer_2_classification_loss', layers[1].metrics['cls'], iter_count)
		# Logs for "YOLO-Layer-3"
		writer.add_scalar('YOLO_Layer_3_Overall_Loss', layers[2].metrics['loss'], iter_count)
		writer.add_scalar('YOLO_Layer_3_x_loss', layers[2].metrics['x'], iter_count)
		writer.add_scalar('YOLO_Layer_3_y_loss', layers[2].metrics['y'], iter_count)
		writer.add_scalar('YOLO_Layer_3_w_loss', layers[2].metrics['w'], iter_count)
		writer.add_scalar('YOLO_Layer_3_h_loss', layers[2].metrics['h'], iter_count)
		writer.add_scalar('YOLO_Layer_3_confidence_loss', layers[2].metrics['conf'], iter_count)
		writer.add_scalar('YOLO_Layer_3_classification_loss', layers[2].metrics['cls'], iter_count)
	
	return loss, outputs

def fit(train_loader, val_loader, model, optimizer, n_epochs):

	print('Training Started\n')

	for epoch in range(n_epochs):
		
		# This is the Training Part of the code
		loss, outputs = run_epoch(train_loader, model, optimizer, epoch_count = epoch)
		# This is the Evaluation Part of the code
		precision, recall, AP, f1_score, ap_class = evaluate(
			model = model,
			val_loader = val_loader,
			iou_thresh = 0.5,
			conf_thresh = 0.5,
			nms_thresh = 0.5,
			img_size = args.inp_img_size,
			)
		
		evaluation_metrics = {
		"val_precision": precision.mean(),
		"val_recall": recall.mean(),
		"val_mAP": AP.mean(),
		"val_f1": f1_score.mean()
		}
		# Logging the above "Evaluation Metrics"
		writer.add_scalar('Validation_Precision', evaluation_metrics['val_precision'], epoch)
		writer.add_scalar('Validation_Recall', evaluation_metrics['val_recall'], epoch)
		writer.add_scalar('Validation_mAP', evaluation_metrics['val_mAP'], epoch)
		writer.add_scalar('Validation_f1_score', evaluation_metrics['val_f1'], epoch)
		# Printing each class AP and then mAP of all classes
		ap_table_data = [['Class Number', 'Class Name', 'Average Precision']]
		
		for class_id in ap_class:
			ap_table_data += [[class_id, class_names[class_id], '%0.5f'%AP[class_id]]]

		ap_table = AsciiTable(ap_table_data)
		print(ap_table.table)
		print("mAP is:", AP.mean())
		torch.save({'model_state_dict': model.cpu().state_dict()}, 'checkpoints/model_epoch_' + str(epoch) + '.pth')

fit(train_loader, val_loader, model, optimizer, n_epochs = args.epochs)