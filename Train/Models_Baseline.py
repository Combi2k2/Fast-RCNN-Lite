import torch.nn as nn

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


class ROI_Layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10 * 10 * 512, 4096)

        self.reg_layer = nn.Linear(4096, 4)
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()    

        self.cls_layer = nn.Linear(4096, 2)
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()
         
    def forward(self, x):
        x = x.view(-1)
        x = nn.ReLU(self.fc1(x))

        pred_score = self.cls_layer(x)
        pred_locs  = self.reg_layer(x)

        return  pred_locs, pred_score