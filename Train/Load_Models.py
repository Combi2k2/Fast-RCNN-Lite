import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import json
import pandas as pd
import cv2
from PIL import Image

# List all the layers of VGG16

from Models_Baseline import *

def load_VGG(device = 'cpu'):
    model = nn.Sequential(
        nn.Conv2d(3,  64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),   nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),   nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),


        nn.Conv2d(64,  128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),


        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),


        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
    )
    model.load_state_dict(torch.load('../models/Backbone.model', map_location = torch.device(device)))

    return  model

def load_RPN(device = 'cpu'):
    RPN_model = RPN_layer().to(device)
    RPN_model.load_state_dict(torch.load('../models/My_RPN.model', map_location = torch.device(device)))

    return  RPN_model

def load_ROI(device = 'cpu'):
    ROI_model = ROI_layer().to(device)
    ROI_model.load_state_dict(torch.load('../models/My_ROI.model', map_location = torch.device(device)))

    return  ROI_model

if __name__ == '__main__':
    fe_extractor = load_VGG('cpu')
    print(fe_extractor)