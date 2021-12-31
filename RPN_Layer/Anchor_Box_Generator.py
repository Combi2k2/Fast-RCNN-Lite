import numpy as np

def Gen_Anchor():
    fe_size = 400 // 8

    ctr_x = np.arange(8, (fe_size + 1) * 8, 8)
    ctr_y = np.arange(8, (fe_size + 1) * 8, 8)

    # coordinates of the 2500 center points to generate anchor boxes
    ctr = np.zeros((fe_size ** 2, 2))

    for i in range(len(ctr_x) * len(ctr_y)):
        x = i // len(ctr_y)
        y = i %  len(ctr_y)

        ctr[i, 0] = ctr_x[x] - 4
        ctr[i, 1] = ctr_y[y] - 4

    # for each of the 2500 anchors, generate 12 anchor boxes
    # 2500 * 9 = 22500 anchor boxes
    ratios = [0.5, 1, 2]
    scales = [4, 8, 16, 32]

    sub_sample = 8
    anchor_boxes = np.zeros(((fe_size * fe_size * 12), 4))
    index = 0

    for c in ctr:
        ctr_x = c[0]
        ctr_y = c[1]

        for i in range(len(ratios)):
            for j in range(len(scales)):
                h = sub_sample * scales[j] * np.sqrt(ratios[i])
                w = sub_sample * scales[j] * np.sqrt(1./ ratios[i])

                anchor_boxes[index, 0] = ctr_x - h / 2.
                anchor_boxes[index, 1] = ctr_y - w / 2.
                anchor_boxes[index, 2] = ctr_x + h / 2.
                anchor_boxes[index, 3] = ctr_y + w / 2.
                index += 1

    index_inside = np.where(
            (anchor_boxes[:, 0] >= 0) &
            (anchor_boxes[:, 1] >= 0) &
            (anchor_boxes[:, 2] < 400) &
            (anchor_boxes[:, 3] < 400)
    )[0]

    return  anchor_boxes, index_inside

if (__name__ == '__main__'):
    anchor_boxes, index_inside = Gen_Anchor()
    valid_anchor_boxes = anchor_boxes[index_inside]

    print(index_inside.shape)
    print(valid_anchor_boxes.shape)