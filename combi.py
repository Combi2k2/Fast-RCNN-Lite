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