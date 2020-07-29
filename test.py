import os
import tqdm
import torch
import argparse
import torchvision
from utils import *
from networks import *
from datasets import *
from terminaltables import AsciiTable
from torch.utils.data import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def evaluate(model, val_loader, iou_thresh, conf_thresh, nms_thresh, img_size):

	model.eval()
	labels = []
	sample_metrics = []  # List of tuples (TP, confs, pred)
	
	for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(val_loader, desc = "Detecting objects")):

		labels += targets[:, 1].tolist()
		targets[:, 2:] = xywh2xyxy(targets[:, 2:])
		targets[:, 2:] *= img_size
		imgs = imgs.type(FloatTensor)

		with torch.no_grad():
			outputs = model(imgs)
			outputs = non_max_suppression(outputs, conf_thresh = conf_thresh, nms_thresh = nms_thresh)

		sample_metrics += get_batch_statistics(outputs, targets, iou_threshold = iou_thresh)

	# Concatenate sample statistics
	true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
	precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

	return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type = int, default = 4, help = "The Image Batch Size")
	parser.add_argument("--network_config", type = str, default = "config/yolov3-custom.cfg", help = "Patch to the file containing the network definition")
	parser.add_argument("--weights_path", type = str, default = "checkpoints/model_epoch_0.pth", help = "Path to the YOLOv3 weights file")
	parser.add_argument("--iou_thresh", type = float, default = 0.5, help = "IOU threshold required to qualify detection as detected")
	parser.add_argument("--conf_thresh", type = float, default = 0.5, help = "Object Confidence Threshold")
	parser.add_argument("--nms_thresh", type = float, default = 0.5, help = "IOU threshold for Non-Maximum Suppression")
	parser.add_argument("--n_cpu", type = int, default = 8, help = "Number of CPU threads to use for Batch Generation")
	parser.add_argument("--inp_img_size", type = int, default = 800, help = "Dimension of input image to the network")
	args = parser.parse_args()

	with open('data/classes.names') as class_name_file:
		names = class_name_file.readlines()

	class_names = []
	for name in names:
		class_names.append(name.rstrip().lstrip())

	# Model Initialization
	model_config = args.network_config
	img_size = args.inp_img_size
	model = Darknet(model_config, img_size)
	
	# Loading the checkpoint weights
	checkpoint = torch.load(args.weights_path)
	model_parameters = checkpoint['model_state_dict']
	model.load_state_dict(model_parameters)
	model.to(device)

	valid_images_name = 'data/valid.txt'
	val_dataset = myDataset(valid_images_name, img_size, multiscale = "False", augmentation = False)
	val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu, collate_fn = val_dataset.collate_fn)

	precision, recall, AP, f1_score, ap_class = evaluate(
		model = model,
		val_loader = val_loader,
		iou_thresh = args.iou_thresh,
		conf_thresh = args.conf_thresh,
		nms_thresh = args.nms_thresh,
		img_size = args.inp_img_size
		)
	
	evaluation_metrics = {
	"val_precision": precision.mean(),
	"val_recall": recall.mean(),
	"val_mAP": AP.mean(),
	"val_f1": f1_score.mean()
	}
	
	# Printing each class AP and then mAP of all classes
	ap_table_data = [['Class Number', 'Class Name', 'Average Precision']]
		
	for class_id in ap_class:
		ap_table_data += [[class_id, class_names[class_id], '%0.5f'%AP[class_id]]]

	ap_table = AsciiTable(ap_table_data)
	print(ap_table.table)
	print("mAP is:", AP.mean())