import tqdm
import torch
import numpy as np

def xywh2xyxy(x):
	y = x.new(x.shape)
	y[..., 0] = x[..., 0] - x[..., 2] / 2
	y[..., 1] = x[..., 1] - x[..., 3] / 2
	y[..., 2] = x[..., 0] + x[..., 2] / 2
	y[..., 3] = x[..., 1] + x[..., 3] / 2
	return y

def ap_per_class(tp, conf, pred_cls, target_cls):
	""" Compute the average precision, given the recall and precision curves.
	Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
	# Arguments
		tp:    True positives (list).
		conf:  Objectness value from 0-1 (list).
		pred_cls: Predicted object classes (list).
		target_cls: True object classes (list).
	# Returns
		The average precision as computed in py-faster-rcnn.
	"""

	# Sort by objectness
	i = np.argsort(-conf)
	tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

	# Find unique classes
	unique_classes = np.unique(target_cls)

	# Create Precision-Recall curve and compute AP for each class
	ap, p, r = [], [], []
	for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
		i = pred_cls == c
		n_gt = (target_cls == c).sum()  # Number of ground truth objects
		n_p = i.sum()  # Number of predicted objects

		if n_p == 0 and n_gt == 0:
			continue
		elif n_p == 0 or n_gt == 0:
			ap.append(0)
			r.append(0)
			p.append(0)
		else:
			# Accumulate FPs and TPs
			fpc = (1 - tp[i]).cumsum()
			tpc = (tp[i]).cumsum()

			# Recall
			recall_curve = tpc / (n_gt + 1e-16)
			r.append(recall_curve[-1])

			# Precision
			precision_curve = tpc / (tpc + fpc)
			p.append(precision_curve[-1])

			# AP from recall-precision curve
			ap.append(compute_ap(recall_curve, precision_curve))

	# Compute F1 score (harmonic mean of precision and recall)
	p, r, ap = np.array(p), np.array(r), np.array(ap)
	f1 = 2 * p * r / (p + r + 1e-16)

	return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
	""" Compute the average precision, given the recall and precision curves.
	Code originally from https://github.com/rbgirshick/py-faster-rcnn.

	# Arguments
		recall:    The recall curve (list).
		precision: The precision curve (list).
	# Returns
		The average precision as computed in py-faster-rcnn.
	"""
	# correct AP calculation
	# first append sentinel values at the end
	mrec = np.concatenate(([0.0], recall, [1.0]))
	mpre = np.concatenate(([0.0], precision, [0.0]))

	# compute the precision envelope
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	# to calculate area under PR curve, look for points
	# where X axis (recall) changes value
	i = np.where(mrec[1:] != mrec[:-1])[0]

	# and sum (\Delta recall) * prec
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap

def get_batch_statistics(outputs, targets, iou_threshold):
	""" Compute true positives, predicted scores and predicted labels per sample """
	batch_metrics = []
	for sample_i in range(len(outputs)):

		if outputs[sample_i] is None:
			continue

		output = outputs[sample_i]
		pred_boxes = output[:, :4]
		pred_scores = output[:, 4]
		pred_labels = output[:, -1]

		true_positives = np.zeros(pred_boxes.shape[0])

		annotations = targets[targets[:, 0] == sample_i][:, 1:]
		target_labels = annotations[:, 0] if len(annotations) else []
		if len(annotations):
			detected_boxes = []
			target_boxes = annotations[:, 1:]

			for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

				# If targets are found break
				if len(detected_boxes) == len(annotations):
					break

				# Ignore if label is not one of the target labels
				if pred_label not in target_labels:
					continue

				iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
				if iou >= iou_threshold and box_index not in detected_boxes:
					true_positives[pred_i] = 1
					detected_boxes += [box_index]
		batch_metrics.append([true_positives, pred_scores, pred_labels])
	return batch_metrics

def bbox_wh_iou(wh1, wh2):
	
	wh2 = wh2.t()
	w1, h1 = wh1[0], wh1[1]
	w2, h2 = wh2[0], wh2[1]
	inter_area = torch.min(w1, w2) * torch.min(h1, h2)
	union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
	
	return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2 = True):
	
	# Returns the IoU of two bounding boxes
	if not x1y1x2y2:
		# Transform from center and width to exact coordinates
		b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
		b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
		b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
		b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
	else:
		# Get the coordinates of bounding boxes
		b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
		b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

	# get the corrdinates of the intersection rectangle
	inter_rect_x1 = torch.max(b1_x1, b2_x1)
	inter_rect_y1 = torch.max(b1_y1, b2_y1)
	inter_rect_x2 = torch.min(b1_x2, b2_x2)
	inter_rect_y2 = torch.min(b1_y2, b2_y2)
	# Intersection area
	inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
		inter_rect_y2 - inter_rect_y1 + 1, min=0
	)
	# Union Area
	b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
	b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

	iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

	return iou

def non_max_suppression(prediction, conf_thresh = 0.5, nms_thresh = 0.5):
	'''
	Removes detections with lower object confidence score than 'conf_thresh' and performs
	Non-Maximum Suppression to further filter detections.
	Returns detections with shape:
		(x1, y1, x2, y2, object_conf, class_score, class_pred)
	'''

	# From (center x, center y, width, height) to (x1, y1, x2, y2)
	prediction[..., :4] = xywh2xyxy(prediction[..., :4])
	output = [None for _ in range(len(prediction))]
	for image_i, image_pred in enumerate(prediction):
		# Filter out confidence scores below threshold
		image_pred = image_pred[image_pred[:, 4] >= conf_thresh]
		# If none are remaining => process next image
		if not image_pred.size(0):
			continue
		# Object confidence times class confidence
		score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
		# Sort by it
		image_pred = image_pred[(-score).argsort()]
		class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
		detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
		# Perform non-maximum suppression
		keep_boxes = []
		while detections.size(0):
			large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thresh
			label_match = detections[0, -1] == detections[:, -1]
			# Indices of boxes with lower confidence scores, large IOUs and matching labels
			invalid = large_overlap & label_match
			weights = detections[invalid, 4:5]
			# Merge overlapping bboxes by order of confidence
			detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
			keep_boxes += [detections[0]]
			detections = detections[~invalid]
		if keep_boxes:
			output[image_i] = torch.stack(keep_boxes)

	return output

def build_targets(pred_boxes, pred_class, targets, anchors, ignore_thresh):

	# Description of the input arguments:

	# pred_boxes -->
	# This will be of shape [batch_size, n_anchors, gs, gs, 4]
	# where,
	# batch_size --> the number of images in a batch
	# n_anchors --> the number of anchors at each scale
	# gs --> the grid size of the feature map dimension

	# pred_class -->
	# This will be of shape [batch_size, n_anchors, gs, gs, 1]

	# targets --> 
	# This will be of shape [n, 6].
	# where, 
	# 'n' will be total number of ground truth objects in the batch of images
	# The 1st column will contain the image number
	# The 2nd column will contain the object class number
	# The 3rd to 6th column will contain the bounding box coordinates i.e., (x_centre, y_centre, width, height)

	# anchors -->
	# This will be of shape [3,2]

	# ignore_thresh
	# It is the IOU threshold to be used during the training

	ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
	FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
	BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor

	nB = pred_boxes.shape[0]
	nA = pred_boxes.shape[1]
	nG = pred_boxes.shape[2]
	nC = pred_class.size(-1)

	# Output tensors
	obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
	noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
	tx = FloatTensor(nB, nA, nG, nG).fill_(0)
	ty = FloatTensor(nB, nA, nG, nG).fill_(0)
	tw = FloatTensor(nB, nA, nG, nG).fill_(0)
	th = FloatTensor(nB, nA, nG, nG).fill_(0)
	tclass = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

	# Map the bounding boxes to the dimensions of the feature map
	target_boxes = targets[:, 2:6] * nG
	gxy = target_boxes[:, :2]
	gwh = target_boxes[:, 2:]
	ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
	# Get the anchors with the best IOU
	best_ious, best_n = ious.max(0)
	b, target_labels = targets[:, :2].long().t()
	gx, gy = gxy.t()
	gw, gh = gwh.t()
	gi, gj = gxy.long().t()

	'''
	gi[gi < 0] = 0	
	gj[gj < 0] = 0
	gi[gi > nG - 1] = nG - 1
	gj[gj > nG - 1] = nG - 1
	'''
	
	# Set the Objectness and the No Objectness Masks
	obj_mask[b, best_n, gj, gi] = 1
	noobj_mask[b, best_n, gj, gi] = 0

	# Set No object mask to zero where IOU exceeds the ignore threshold value
	for i, anchor_ious in enumerate(ious.t()):
		noobj_mask[b[i], anchor_ious > ignore_thresh, gj[i], gi[i]] = 0

	tx[b, best_n, gj, gi] = gx - gx.floor()
	ty[b, best_n, gj, gi] = gy - gy.floor()
	tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
	th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
	tclass[b, best_n, gj, gi, target_labels] = 1
	tconf = obj_mask.float()

	return obj_mask, noobj_mask, tx, ty, tw, th, tclass, tconf	