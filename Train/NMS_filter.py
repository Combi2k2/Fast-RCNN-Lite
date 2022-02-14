import torch
import numpy as np

def Proposed_Regions(base_boxes, pred_anchor_locs_numpy):
    base_height = base_boxes[:, 2] - base_boxes[:, 0]
    base_width  = base_boxes[:, 3] - base_boxes[:, 1]

    base_ctr_x = base_boxes[:, 0] + 0.5 * base_height
    base_ctr_y = base_boxes[:, 1] + 0.5 * base_width

    # The 30000 anchor boxes location and labels predicted by RPN (convert to numpy)
    # format = (dy, dx, dh, dw)
    dx = pred_anchor_locs_numpy[:, 0::4] # dx
    dy = pred_anchor_locs_numpy[:, 1::4] # dy
    dh = pred_anchor_locs_numpy[:, 2::4] # dh
    dw = pred_anchor_locs_numpy[:, 3::4] # dw
    
    # ctr_x = dx predicted by RPN * anchor_h + anchor_cx
    # ctr_y similar
    # h = exp(dh predicted by RPN) * anchor_h
    # w similar
    ctr_x = dx * base_height[:, np.newaxis] + base_ctr_x[:, np.newaxis]
    ctr_y = dy * base_width[:, np.newaxis]  + base_ctr_y[:, np.newaxis]
    
    h = np.exp(dh) * base_height[:, np.newaxis]
    w = np.exp(dw) * base_width[:, np.newaxis]

    # the final locations of the predicted bounding boxes have the form (x1, y1, x2, y2)
    roi = np.zeros(pred_anchor_locs_numpy.shape, dtype = pred_anchor_locs_numpy.dtype)
    roi[:, 0::4] = ctr_x - 0.5 * h
    roi[:, 1::4] = ctr_y - 0.5 * w
    roi[:, 2::4] = ctr_x + 0.5 * h
    roi[:, 3::4] = ctr_y + 0.5 * w

    # trim the boxes which cover the area outside of the image
    roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, 400)
    roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, 400)

    return  roi

nms_thresh = 0.7  # non-maximum supression (NMS)
min_size   = 8


def NMS_filter(base_boxes, pred_locs, pred_score):
    pred_locs  = pred_locs.contiguous().view(-1, 4)
    pred_score = pred_score.contiguous().view(-1, 2)

    pred_anchor_locs_numpy  = pred_locs.cpu().data.numpy()
    pred_anchor_score_numpy = pred_score.cpu().data.numpy()[:, 1]

    roi = Proposed_Regions(base_boxes, pred_anchor_locs_numpy)

    # elimiate proposed boxes which are too small
    hs = roi[:, 2] - roi[:, 0]
    ws = roi[:, 3] - roi[:, 1]

    keep = np.where((hs >= min_size) & 
                    (ws >= min_size))[0]
    
    roi = roi[keep]

    # start NMS filter
    score = pred_anchor_score_numpy[keep]
    order = score.ravel().argsort()[::-1]
    order = order[:1000]

    roi = roi[order, :]

    # Take all the roi boxes [roi_array]
    x1 = roi[:, 0]; y1 = roi[:, 1]
    x2 = roi[:, 2]; y2 = roi[:, 3]

    # Find the areas of all the boxes [roi_area]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    #Take the indexes of order the probability score in descending order 
    order = order.argsort()[::-1]
    keep = []

    while (order.size > 0):
        i = order[0] #take the 1st elt in order and append to keep 
        keep.append(i)

        inner_x1 = np.maximum(x1[i], x1[order[1:]])
        inner_y1 = np.maximum(y1[i], y1[order[1:]])
        inner_x2 = np.minimum(x2[i], x2[order[1:]])
        inner_y2 = np.minimum(y2[i], y2[order[1:]])

        inner_h = np.maximum(0.0, inner_x2 - inner_x1 + 1)
        inner_w = np.maximum(0.0, inner_y2 - inner_y1 + 1)

        inter = inner_h * inner_w
        outer = areas[i] + areas[order[1:]] - inter

        ious = inter / outer
        inds = np.where(ious < nms_thresh)[0]

        order = order[inds + 1]
    
    return  roi[keep]# the final region proposals