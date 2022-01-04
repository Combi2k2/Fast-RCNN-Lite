import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import json
import cv2

def Load_Annotations(base_path):
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

    return  [cv2.imread(f'{base_path}/{name}') for name in pic_names], pic_boxes

'''
This helps rebalancing the label of the dataset (which initially too bias to negative label)
'''
def label_balancing(label, num_classes = 2):
    pos_index = np.where(label == 1)[0]
    neg_index = np.where(label == 0)[0]

    n_pos = 0.5 * label.shape[0]
    n_neg = np.sum(label == 1)

    if len(pos_index) > n_pos:  label[np.random.choice(pos_index, size = (len(pos_index) - n_pos), replace = False)] = -1
    if len(neg_index) > n_neg:  label[np.random.choice(neg_index, size = (len(neg_index) - n_neg), replace = False)] = -1

'''
calculate the ground truth value of (dx, dy, dh, dw) (This will be use twice so i wrote it)
'''
def calc_gt_locs(regions, gt_bbox):
    height = regions[:, 2] - regions[:, 0]
    width  = regions[:, 3] - regions[:, 1]
    ctr_x  = regions[:, 0] + 0.5 * height
    ctr_y  = regions[:, 1] + 0.5 * width

    # groundtruth box which correspond this valid anchor box h, w, cx, cy 
    base_height = gt_bbox[:, 2] - gt_bbox[:, 0]
    base_width  = gt_bbox[:, 3] - gt_bbox[:, 1]
    base_ctr_x  = gt_bbox[:, 0] + 0.5 * base_height
    base_ctr_y  = gt_bbox[:, 1] + 0.5 * base_width

    # valid anchor boxes loc = (y-ya/ha), (x-xa/wa), log(h/ha), log(w/wa)
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps) #make height != 0 by let its minimum value be eps
    width  = np.maximum(width,  eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height

    dh = np.log(base_height / height)
    dw = np.log(base_width  / width)

    return  np.vstack((dx, dy, dh, dw)).transpose()

'''
START creating my custom dataset
'''

from My_Transforms.Image_Transform import Image_Prep
from My_Transforms.Boxes_Transform import Annotations_Prep

class RPN_Dataset(Dataset):
    def __init__(self, base_path):
        super().__init__()

        images, bboxes = Load_Annotations(base_path)

        self.images = images
        self.bboxes = bboxes

        self.n_samples = len(images)
    
    def __getitem__(self, index):
        image  = self.images[index]
        target = self.bboxes[index]
        
        return  Image_Prep(image), Annotations_Prep(target)
    
    def __len__(self):
        return  self.n_samples

class ROI_Dataset(Dataset):
    def __init__(self, base_path, device = 'cpu'):
        super().__init__()

        images, bboxes = Load_Annotations(base_path)
        self.input  = []
        self.target = []


        import Load_Models
        fe_extractor = Load_Models.loload_VGG(device)
        RPN_model    = Load_Models.load_RPN(device)
        

        for img, bbox in zip(images, bboxes):
            img = Image_Prep(img)
            img = img.unsqueeze(0).to(device)

            bbox = np.array(bbox)

            with torch.no_grad():
                outmap = fe_extractor(img)
                pred_locs, pred_score = RPN_model(outmap)

                rois = NMS_filter(pred_locs, pred_score)
            
            ious = np.zeros((len(rois), len(bbox)), dtype = np.float32)

            for index in range(len(rois) * len(bbox)):
                i = index // len(bbox)
                j = index  % len(bbox)

                ious[i, j] = calc_iou(rois[i], bbox[j])
                
            gt_assignment = ious.argmax(axis = 1)
            gt_roi_label = np.ones(gt_assignment.shape)

            max_iou = ious.max(axis = 1)

            # assign if the proposed region has a chesspiece or not
            pos_index = np.where(max_iou >= 0.7)[0]
            neg_index = np.where(max_iou <  0.3)[0]
            
            keep_index = np.append(pos_index, neg_index)

            gt_roi_labels = gt_roi_label[keep_index]
            gt_roi_labels[pos_index.size:] = 0  # negative labels --> 0

            label_balancing(gt_roi_labels)

            # calculate the groundtruth locs for proposed regions:
            sample_roi = rois[keep_index]
            bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]

            gt_roi_locs = calc_gt_locs(sample_roi, bbox_for_sampled_roi)

            rois = torch.from_numpy(sample_roi).float()
            rois = rois / 8.0 #Downsampling ratio
            rois = rois.long()

            if (rois.shape[0] != keep_index.shape[0]):
                print("BUG detected")
            
            for i in range(rois.shape[0]):
                x1, y1, x2, y2 = rois[i]
                img = outmap[0][..., x1: x2 + 1, y1: y2 + 1]
                tmp = adaptive_max_pool(img)

                self.input.append(tmp[0])
                self.target.append((gt_roi_labels[i], gt_roi_locs[i]))

        self.n_samples = len(self.input)

    def __getitem__(self, index):
        return  self.input[index], self.target[index]

    def __len__(self):
        return  self.n_samples