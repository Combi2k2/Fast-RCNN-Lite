import numpy as np
from Anchor_Box_Generator import Gen_Anchor

anchor_boxes, index_inside = Gen_Anchor()
valid_anchor_boxes = anchor_boxes[index_inside]

def Annotations_Prep(bbox):
    bbox = np.array(bbox)
    ious = np.zeros((len(valid_anchor_boxes), len(bbox)))
    
    for i in range(valid_anchor_boxes.shape[0]):
        xa1, ya1, xa2, ya2 = valid_anchor_boxes[i]
        
        anchor_area = (xa2 - xa1) * (ya2 - ya1)
        
        for j in range(len(bbox)):
            xb1, yb1, xb2, yb2 = bbox[j]
            
            box_area = (yb2 - yb1) * (xb2 - xb1)

            inner_x1 = max(xa1, xb1);   inner_x2 = min(xa2, xb2)
            inner_y1 = max(ya1, yb1);   inner_y2 = min(ya2, yb2)

            if (inner_x1 >= inner_x2):  continue
            if (inner_y1 >= inner_y2):  continue

            inner_area = (inner_y2 - inner_y1) * (inner_x2 - inner_x1)
            ious[i, j] = inner_area / (anchor_area + box_area - inner_area)

    gt_max_ious = ious.max(axis = 0)
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]
    
    argmax_ious = ious.argmax(axis = 1)
    max_ious = ious[np.arange(len(index_inside)), argmax_ious]

    """
    15280 valid anchor boxes with label:
    1: human
    0: background
    -1: ignored
    """
    label = np.empty((len(index_inside), ), dtype = np.int32)
    label.fill(-1)
    
    # Use iou to assign 1 (objects) to two kind of anchors 
    # a) The anchors with the highest iou overlap with a ground-truth-box
    # - An anchor that has an IoU overlap higher than 0.7 with ground-truth box is assigned label "foreground"
    # - An anchor that has an IoU overlap lower  than 0.3 with ground-truth box is assigned label "background"
    pos_iou_threshold = 0.7
    neg_iou_threshold = 0.3
    label[gt_argmax_ious] = 1
    label[max_ious > pos_iou_threshold] = 1
    label[max_ious < neg_iou_threshold] = 0
    
    pos_index = np.where(label == 1)[0]
    neg_index = np.where(label == 0)[0]

    n_pos = 0.5 * ious.shape[0]
    n_neg = np.sum(label == 1)

    if len(pos_index) > n_pos:  label[np.random.choice(pos_index, size = (len(pos_index) - n_pos), replace = False)] = -1
    if len(neg_index) > n_neg:  label[np.random.choice(neg_index, size = (len(neg_index) - n_neg), replace = False)] = -1
    
    
    # For each valid anchor box, find the groundtruth object which has max_iou 
    max_iou_bbox = bbox[argmax_ious]

    # valid anchor boxes h, w, cx, cy 
    height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
    width  = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
    ctr_x  = valid_anchor_boxes[:, 0] + 0.5 * height
    ctr_y  = valid_anchor_boxes[:, 1] + 0.5 * width

    # groundtruth box which correspond this valid anchor box h, w, cx, cy 
    base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
    base_width  = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
    base_ctr_x  = max_iou_bbox[:, 0] + 0.5 * base_height
    base_ctr_y  = max_iou_bbox[:, 1] + 0.5 * base_width

    # valid anchor boxes loc = (y-ya/ha), (x-xa/wa), log(h/ha), log(w/wa)
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps) #make height != 0 by let its minimum value be eps
    width  = np.maximum(width,  eps)
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width  / width)
    
    anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
        
    anchor_labels = np.empty((len(anchor_boxes),), dtype = label.dtype)
    anchor_labels.fill(-1)
    anchor_labels[index_inside] = label

    anchor_locations = np.empty((len(anchor_boxes),) + anchor_boxes.shape[1:], dtype = anchor_locs.dtype)
    anchor_locations.fill(0)
    anchor_locations[index_inside, :] = anchor_locs
    
    return  anchor_labels, anchor_locations