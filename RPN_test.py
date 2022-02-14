import matplotlib.pyplot as plt
import numpy  as np
import cv2
import os
import json

import torch
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid


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

for i, fe in enumerate(layer):
    print(f'Feature {i + 1}: {fe}')

# extract the feature maps of size 50 x 50 (apply MaxPool2d 3 times)
layer = layer[: 23]

# Convert this list into a Sequential module
fe_extractor = nn.Sequential(*layer).to(device)


#load my pretrained model

from Train.Anchor_Box_Generator import Gen_Anchor
from Train.NMS_filter import NMS_filter
from Train.Load_Models import *

from Train.My_Transforms.Image_Transform import Image_Prep
from Train.My_Transforms.Boxes_Transform import Annotations_Prep

anchor_boxes = Gen_Anchor()

model = RPN_layer()
model.load_state_dict(torch.load('mask-classifier.model', map_location = torch.device(device)))

RPN_model = load_RPN()

with torch.no_grad():
    img = cv2.imread('/content/test/a3863d0be6002c21b20ac88817b2c56f_jpg.rf.0413d5178136ace55f588df9556c060a.jpg')

    # calculate predicted label and locs
    img_clone = Image_Prep(img)
    img_clone = img_clone.unsqueeze(0).to(device)

    outmap = fe_extractor(img_clone)
    pred_locs, pred_score = RPN_model(outmap)
    
    pred_locs = pred_locs.permute(0, 2, 3, 1)
    pred_score = pred_score.permute(0, 2, 3, 1)

    rois = NMS_filter(anchor_boxes, pred_locs, pred_score)

    # show the bounding box of predicted boxes:
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, dsize = (400, 400), interpolation = cv2.INTER_AREA)


    #show the most potential regions
    fig, ax = plt.subplots(1, 3, figsize=(15,15))
    fig.tight_layout()

    img_clone = img.copy()

    for i in range(rois.shape[0] // 3):
        x1, y1, x2, y2 = rois[i]
        cv2.rectangle(img_clone, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    ax[0].imshow(img_clone)

    #show the regions which are less likely to contain chesspieces
    img_clone = img.copy()

    for i in range(rois.shape[0] // 3, 2 * rois.shape[0] // 3):
        x1, y1, x2, y2 = rois[i]
        cv2.rectangle(img_clone, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    ax[1].imshow(img_clone)

    #show the least potential regions
    img_clone = img.copy()

    for i in range(2 * rois.shape[0] // 3, rois.shape[0]):
        x1, y1, x2, y2 = rois[i]
        cv2.rectangle(img_clone, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    ax[2].imshow(img_clone)