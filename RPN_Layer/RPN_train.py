import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import cv2
from PIL import Image

from Image_Transform import Image_Prep
from Boxes_Transform import Annotations_Prep

#Setting up my device
if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print(device)

# List all the layers of VGG16
model = torchvision.models.vgg16(pretrained = True).to(device)
layer = list(model.features)


'''
---Loading Pretrained Model---
'''
# extract the feature maps of size 50 x 50 (apply MaxPool2d 3 times)
layer = layer[: 23]
# Convert this list into a Sequential module
fe_extractor = nn.Sequential(*layer).to(device)


MyTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([400, 400])
]) # Defing PyTorch Transform

'''
---Custom Dataset---
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ChessDataset(Dataset):
    def __init__(self, base_path):
        super().__init__()
        
        data = None
        
        with open(f'{base_path}/_annotations.coco.json', 'r') as f:
            data = json.load(f)
        
        images = data["images"]
        bboxes = data["annotations"]
        
        pic_names = ["" for i in range(len(images))]
        pic_boxes = [[] for i in range(len(images))]
        
        H_ratio = 400/416.
        W_ratio = 400/416.
        
        for img in images:  pic_names[img["id"]] = img["file_name"]
        for box in bboxes:
            pic_name = box["image_id"]
            box_locs = box["bbox"]
            
            x1, y1, h, w = box_locs
            x2 = x1 + h
            y2 = y1 + w
            
            x1 = x1 * H_ratio;  y1 = y1 * W_ratio
            x2 = x2 * H_ratio;  y2 = y2 * W_ratio
            
            pic_boxes[pic_name].append((x1, y1, x2, y2))
            
        self.images = [cv2.imread(f'{base_path}/{name}') for name in pic_names]
        self.bboxes = pic_boxes

        self.n_samples = len(images)
    
    def __getitem__(self, index):
        image  = self.images[index]
        target = self.bboxes[index]

        return  Image_Prep(image), Annotations_Prep(target)
    
    def __len__(self):
        return  self.n_samples

train_ds = ChessDataset('../Chess Pieces.v24-416x416_aug.coco/train')
valid_ds = ChessDataset('../Chess Pieces.v24-416x416_aug.coco/valid')
test_ds = ChessDataset('../Chess Pieces.v24-416x416_aug.coco/test')

num_epochs = 10
batch_size = 1
learning_rate = 0.01

train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 2, pin_memory = True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle = True, num_workers = 2, pin_memory = True)
test_dl  = DataLoader(test_ds,  batch_size, shuffle = True, num_workers = 2, pin_memory = True)

'''
---Region Proposal Network---
'''
n_anchor = 12 # number of anchor boxes having the center being the current point

class RPN_layer(nn.Module):
    def __init__(self):
        super().__init__()
        inp_channels = 512
        mid_channels = 512
        
        self.conv1 = nn.Conv2d(inp_channels, mid_channels, 3, 1, 1).to(device)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()

        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0).to(device)
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()

        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0).to(device) ## I will use softmax here. you can equally use sigmoid if u replace 2 with 1.
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv1(x)
        
        pred_anchor_locs  = self.reg_layer(x)
        pred_anchor_label = self.cls_layer(x)
        
        return  pred_anchor_locs, pred_anchor_label

'''
---Initialize my model---
'''
model = RPN_layer().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

'''
---Training Phase---
'''
for epoch in range(num_epochs):
    rpn_lambda = 10.
    batch_index = 1
    
    for (images, target) in train_dl:
        images = images.to(device)

        outmap = fe_extractor(images)

        anchor_label = target[0].to(device).contiguous().view(-1)
        anchor_locs  = target[1].to(device).contiguous().view(-1, 4)
        
        #forward step
        pred_locs, pred_label = model(outmap)

        pred_locs  = pred_locs.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        pred_label = pred_label.permute(0, 2, 3, 1).contiguous().view(-1, 2)

        pos  = anchor_label > 0
        mask = pos.unsqueeze(1).expand_as(pred_locs)

        pred_locs   = pred_locs[mask]
        anchor_locs = anchor_locs[mask]
        
        rpn_cls_loss = F.cross_entropy(pred_label, anchor_label.long(), ignore_index = -1)
        rpn_loc_loss = F.smooth_l1_loss(pred_locs, anchor_locs)
        
        rpn_total_loss = rpn_cls_loss + rpn_loc_loss * rpn_lambda
        
        optimizer.zero_grad(); rpn_total_loss.backward()
        optimizer.step()

        if (batch_index % 50 == 0):
            print(f'Epoch {epoch + 1}/{num_epochs}, batch {batch_index}: train_loss = {rpn_total_loss}')
        
        batch_index += 1


'''
---Saving model---
'''
