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
from RPN_Layer.RPN_train import RPN_layer
from RPN_Layer.Anchor_Box_Generator import Gen_Anchor
from RPN_Layer.Image_Transform import Image_Prep

anchor_boxes, index_inside = Gen_Anchor()
valid_anchor_boxes = anchor_boxes[index_inside]

anc_height = anchor_boxes[:, 2] - anchor_boxes[:, 0]
anc_width  = anchor_boxes[:, 3] - anchor_boxes[:, 1]
anc_ctr_y = anchor_boxes[:, 0] + 0.5 * anc_height
anc_ctr_x = anchor_boxes[:, 1] + 0.5 * anc_width



model = RPN_layer()
model.load_state_dict(torch.load('mask-classifier.model', map_location = torch.device(device)))

with torch.no_grad():
    img = cv2.imread('Chess Pieces.v24-416x416_aug.coco/test/0b47311f426ff926578c9d738d683e76_jpg.rf.40183eae584a653181bbd795ba3c353f.jpg')

    img_clone = Image_Prep(img)
    img_clone = img_clone.unsqueeze(0).to(device)

    outmap = fe_extractor(img_clone)
    pred_locs, pred_label = model(outmap)

    pred_locs  = pred_locs.permute(0, 2, 3, 1).contiguous().view(-1, 4)
    pred_label = pred_label.permute(0, 2, 3, 1).contiguous().view(-1, 2)

    pred_anchor_locs_numpy = pred_locs.cpu().data.numpy()

    # The 30000 anchor boxes location and labels predicted by RPN (convert to numpy)
    # format = (dy, dx, dh, dw)
    dy = pred_anchor_locs_numpy[:, 0::4] # dy
    dx = pred_anchor_locs_numpy[:, 1::4] # dx
    dh = pred_anchor_locs_numpy[:, 2::4] # dw
    dw = pred_anchor_locs_numpy[:, 3::4] # dh

    # ctr_y = dy predicted by RPN * anchor_h + anchor_cy
    # ctr_x similar
    # h = exp(dh predicted by RPN) * anchor_h
    # w similar
    ctr_x = dx * anc_height[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    ctr_y = dy * anc_width[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    
    h = np.exp(dh) * anc_height[:, np.newaxis]
    w = np.exp(dw) * anc_width[:, np.newaxis]
    print(w.shape)

    _, predictions = torch.max(pred_label, 1)

    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, dsize = (400, 400), interpolation = cv2.INTER_AREA)

    pos = np.where(predictions > 0)[0]

    print(pos.shape[0])
    pos = np.random.choice(pos, size = 100)

    print(pos)

    for i in pos:
        x1 = int(ctr_x[i] - 0.5 * h[i])
        y1 = int(ctr_y[i] - 0.5 * w[i])

        x2 = int(ctr_x[i] + 0.5 * h[i])
        y2 = int(ctr_y[i] + 0.5 * w[i])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

plt.figure(figsize = (10, 10))
plt.imshow(img)