import torch.nn as nn
import torch.nn.functional as F

n_anchor = 12 # number of anchor boxes having the center being the current point

class RPN_layer(nn.Module):
    def __init__(self):
        super().__init__()
        inp_channels = 512
        mid_channels = 512
        
        self.conv1 = nn.Conv2d(inp_channels, mid_channels, 3, 1, 1)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()

        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()

        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) ## I will use softmax here. you can equally use sigmoid if u replace 2 with 1.
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv1(x)
        
        pred_anchor_locs  = self.reg_layer(x)
        pred_anchor_label = self.cls_layer(x)
        
        return  pred_anchor_locs, pred_anchor_label


class ROI_Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(512 * 7 * 7, 128)

        self.reg_layer = nn.Linear(128, 4)  
        self.cls_layer = nn.Linear(128, 2)
         
    def forward(self, x):
        out = x.view(-1, 512 * 7 * 7)
        out = F.relu(self.fc1(out))

        pred_locs  = self.reg_layer(out)
        pred_score = self.cls_layer(out)
        
        return  pred_locs, pred_score