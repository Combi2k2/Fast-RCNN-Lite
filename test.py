import matplotlib.pyplot as plt
import numpy  as np
import cv2
import os
import json

base_path_train = 'Chess Pieces.v24-416x416_aug.coco/train'
base_path_valid = 'Chess Pieces.v24-416x416_aug.coco/valid'
base_path_test  = 'Chess Pieces.v24-416x416_aug.coco/test'
data = dict()

with open(base_path_train + '/_annotations.coco.json', 'r') as f:
    data = json.load(f)

images = data["images"]
bboxes = data["annotations"]

file_names = [None] * len(images)
file_boxes = [[] for i in range(len(images))]

for img in images:  file_names[img["id"]] = img["file_name"]
for box in bboxes:  file_boxes[box["image_id"]].append(box["bbox"])

img_dir = '0b47311f426ff926578c9d738d683e76_jpg.rf.40183eae584a653181bbd795ba3c353f.jpg'
img = cv2.imread(base_path_train + '/' + img_dir)

idx = file_names.index(img_dir)
print(idx)

for box in file_boxes[idx]:
    x1, y1, h, w = box
    x2 = int(x1 + h)
    y2 = int(y1 + w)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

plt.imshow(img)
plt.show()