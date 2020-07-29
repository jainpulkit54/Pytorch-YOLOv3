import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import build_targets, non_max_suppression
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_model_config(path):
	# Parses the yolo-v3 layer configuration file and returns module definitions
	file = open(path, 'r')
	lines = file.read().split('\n')
	lines = [x for x in lines if x and not x.startswith('#')]
	lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
	module_defs = []
	for line in lines:
		if line.startswith('['): # This marks the start of a new block
			module_defs.append({})
			module_defs[-1]['type'] = line[1:-1].rstrip()
			if module_defs[-1]['type'] == 'convolutional':
				module_defs[-1]['batch_normalize'] = 0
		else:
			key, value = line.split("=")
			value = value.strip()
			module_defs[-1][key.rstrip()] = value.strip()

	return module_defs

def create_modules(module_defs):

	# Constructs module list of layer blocks from module configuration in module_defs
	hyperparams = module_defs.pop(0)
	output_filters = [int(hyperparams["channels"])]
	module_list = nn.ModuleList()
	for module_i, module_def in enumerate(module_defs):
		modules = nn.Sequential()

		if module_def["type"] == "convolutional":
			bn = int(module_def["batch_normalize"])
			filters = int(module_def["filters"])
			kernel_size = int(module_def["size"])
			pad = (kernel_size - 1) // 2
			modules.add_module(
				f"conv_{module_i}",
				nn.Conv2d(
					in_channels = output_filters[-1],
					out_channels = filters,
					kernel_size = kernel_size,
					stride = int(module_def["stride"]),
					padding = pad,
					bias = not bn,
				),
			)
			if bn:
				modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum = 0.9, eps = 1e-5))
			if module_def["activation"] == "leaky":
				modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

		elif module_def["type"] == "maxpool":
			kernel_size = int(module_def["size"])
			stride = int(module_def["stride"])
			if kernel_size == 2 and stride == 1:
				modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
			maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) // 2))
			modules.add_module(f"maxpool_{module_i}", maxpool)

		elif module_def["type"] == "upsample":
			upsample = Upsample(scale_factor = int(module_def["stride"]), mode = "nearest")
			modules.add_module(f"upsample_{module_i}", upsample)

		elif module_def["type"] == "route":
			layers = [int(x) for x in module_def["layers"].split(",")]
			filters = sum([output_filters[1:][i] for i in layers])
			modules.add_module(f"route_{module_i}", EmptyLayer())

		elif module_def["type"] == "shortcut":
			filters = output_filters[1:][int(module_def["from"])]
			modules.add_module(f"shortcut_{module_i}", EmptyLayer())

		elif module_def["type"] == "yolo":
			anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
			# Extract anchors
			anchors = [int(x) for x in module_def["anchors"].split(",")]
			anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in anchor_idxs]
			num_classes = int(module_def["classes"])
			# Define detection layer
			yolo_layer = YOLOLayer(anchors, num_classes)
			modules.add_module(f"yolo_{module_i}", yolo_layer)
		# Register module list and number of output filters
		module_list.append(modules)
		output_filters.append(filters)

	return module_list

class Upsample(nn.Module):
	""" nn.Upsample is deprecated """

	def __init__(self, scale_factor, mode = "nearest"):
		super(Upsample, self).__init__()
		self.scale_factor = scale_factor
		self.mode = mode

	def forward(self, x):
		x = F.interpolate(x, scale_factor = self.scale_factor, mode = self.mode)
		return x

# The below class acts as the placeholder for "route" and "shortcut" layers
class EmptyLayer(nn.Module):

	def __init__(self):
		super(EmptyLayer, self).__init__()

# The below class is the "Detection" Layer
class YOLOLayer(nn.Module):

	def __init__(self, anchors, num_classes):
		
		super(YOLOLayer, self).__init__()
		self.anchors = anchors
		self.num_anchors = len(anchors) # It will be 3 for each Scale
		self.num_classes = num_classes
		self.ignore_thresh = 0.5 # It is the IOU threshold to be used during the training
		self.mse_loss = nn.MSELoss(reduction = 'mean')
		self.bce_loss = nn.BCELoss(reduction = 'mean')
		self.obj_scale = 1
		self.noobj_scale = 100
		self.metrics = {}
		self.grid_size = 0  # grid size

	def compute_grid_offsets(self, grid_size, img_dim):
		
		FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
		self.grid_size = grid_size
		self.img_dim = img_dim
		g = self.grid_size
		self.stride = self.img_dim / self.grid_size
		# Calculate offsets for each grid
		self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
		self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
		self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
		self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
		self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

	def forward(self, x, targets = None, img_dim = 416):

		# Tensors for cuda support
		FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
		LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
		ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

		batch_size = x.shape[0]
		grid_size = x.shape[2]
		prediction = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
		
		# Get outputs
		tx_hat = torch.sigmoid(prediction[:,:,:,:,0]) # For Center-x, we apply sigmoid on prediction to ensure value is between 0 and 1
		ty_hat = torch.sigmoid(prediction[:,:,:,:,1]) # For Center-y, we apply sigmoid on prediction to ensure value is between 0 and 1
		tw_hat = prediction[:,:,:,:,2] # Width
		th_hat = prediction[:,:,:,:,3] # Height
		pred_conf = torch.sigmoid(prediction[:,:,:,:,4]) # Object Confidence
		pred_class = torch.sigmoid(prediction[:,:,:,:,5:]) # Class Prediction Probability

		# If grid size does not match current we compute new offsets
		if grid_size != self.grid_size:
			self.compute_grid_offsets(grid_size, img_dim)

		# The Log-Space Transformations (Adding the offsets and scaling with the anchors)
		pred_boxes = FloatTensor(prediction[:,:,:,:,:4].shape)
		pred_boxes[:,:,:,:,0] = tx_hat + self.grid_x
		pred_boxes[:,:,:,:,1] = ty_hat + self.grid_y
		pred_boxes[:,:,:,:,2] = torch.exp(tw_hat) * self.anchor_w
		pred_boxes[:,:,:,:,3] = torch.exp(th_hat) * self.anchor_h
		output = torch.cat(
			(
				pred_boxes.view(batch_size, -1, 4) * self.stride,
				pred_conf.view(batch_size, -1, 1),
				pred_class.view(batch_size, -1, self.num_classes),
			),
			-1,
		)

		if targets is not None:

			obj_mask, noobj_mask, tx, ty, tw, th, tclass, tconf = build_targets(
				pred_boxes = pred_boxes,
				pred_class = pred_class,
				targets = targets,
				anchors = self.scaled_anchors,
				ignore_thresh = self.ignore_thresh
				)

			loss_x = self.mse_loss(tx_hat[obj_mask], tx[obj_mask])
			loss_y = self.mse_loss(ty_hat[obj_mask], ty[obj_mask])
			loss_w = self.mse_loss(tw_hat[obj_mask], tw[obj_mask])
			loss_h = self.mse_loss(th_hat[obj_mask], th[obj_mask])
			loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
			loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
			loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
			loss_class = self.bce_loss(pred_class[obj_mask], tclass[obj_mask])
			total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_class

			# Metrics
			conf_obj = pred_conf[obj_mask].mean()
			conf_noobj = pred_conf[noobj_mask].mean()

			self.metrics = {
				"loss": total_loss.cpu().item(),
				"x": loss_x.cpu().item(),
				"y": loss_y.cpu().item(),
				"w": loss_w.cpu().item(),
				"h": loss_h.cpu().item(),
				"conf": loss_conf.cpu().item(),
				"cls": loss_class.cpu().item(),
				"conf_obj": conf_obj.cpu().item(),
				"conf_noobj": conf_noobj.cpu().item(),
				"grid_size": grid_size,
			}
			return output, total_loss
		else:
			return output, 0

class Darknet(nn.Module):

	def __init__(self, config_path, img_size):
		
		super(Darknet, self).__init__()
		self.module_defs = parse_model_config(config_path)
		self.module_list = create_modules(self.module_defs)
		self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
		self.img_size = img_size
		self.seen = 0
		self.header_info = np.array([0, 0, 0, self.seen, 0], dtype = np.int32)

	def forward(self, x, targets = None):
		
		loss = 0
		img_dim = x.shape[2]
		layer_outputs, yolo_outputs = [], []
		for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
			if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
				x = module(x)
			elif module_def["type"] == "route":
				x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
			elif module_def["type"] == "shortcut":
				layer_i = int(module_def["from"])
				x = layer_outputs[-1] + layer_outputs[layer_i]
			elif module_def["type"] == "yolo":
				x, layer_loss = module[0](x, targets, img_dim)
				loss += layer_loss
				yolo_outputs.append(x)
			layer_outputs.append(x)
		yolo_outputs = torch.cat(yolo_outputs, 1).cpu()
		
		if targets is None:
			return yolo_outputs
		else:
			return loss, yolo_outputs

	def load_darknet_weights(self, weights_path):
		# Parses and loads the pretrained "Darknet-53" weights

		# Open the weights file
		with open(weights_path, "rb") as f:
			header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
			self.header_info = header  # Needed to write header when saving weights
			self.seen = header[3]  # number of images seen during training
			weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

		# Establish cutoff for loading backbone weights
		cutoff = None
		if "darknet53.conv.74" in weights_path:
			cutoff = 75

		ptr = 0
		for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
			if i == cutoff:
				break
			if module_def["type"] == "convolutional":
				conv_layer = module[0]
				if module_def["batch_normalize"]:
					# Load BN bias, weights, running mean and running variance
					bn_layer = module[1]
					num_b = bn_layer.bias.numel()  # Number of biases
					# Bias
					bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
					bn_layer.bias.data.copy_(bn_b)
					ptr += num_b
					# Weight
					bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
					bn_layer.weight.data.copy_(bn_w)
					ptr += num_b
					# Running Mean
					bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
					bn_layer.running_mean.data.copy_(bn_rm)
					ptr += num_b
					# Running Var
					bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
					bn_layer.running_var.data.copy_(bn_rv)
					ptr += num_b
				else:
					# Load conv. bias
					num_b = conv_layer.bias.numel()
					conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
					conv_layer.bias.data.copy_(conv_b)
					ptr += num_b
				# Load conv. weights
				num_w = conv_layer.weight.numel()
				conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
				conv_layer.weight.data.copy_(conv_w)
				ptr += num_w