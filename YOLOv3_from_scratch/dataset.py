import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)
ImageFile.LOAD_TRUNCATED_IMAGES = True
class YOLODataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,label_dir,
            anchors,
            images_size = 416,
            S = [13,26,52],
            C = 20,
            transform = None,

    ) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # for all 3 scales
        self.num_anchors = self.anchors.shape[0] # 9
        self.num_anchors_per_scale =  self.num_anchors // 3 # 3
        self.C = C
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # np.roll, [label,x,y,w,h] -> [x,y,w,h,label] 
        bboxes = np.roll(np.loadtxt(fname = label_path, delimiter = " ", ndmin=2), 4, axis = 1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            # if there is a rotation, you should rotate the bbox too
            augmentations = self.transform(image = image, bboxes = bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']
        #[3,S,S,6]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6))for S in self.S]#[[0,1],x,y,w,h,class]
        print('targets',targets[0].shape)
        for box in bboxes:

            # 9 anchors which matches best
            # iou_anchors tensor([0.1020, 0.3021, 0.7510, 0.0174, 0.0273, 0.0672, 0.0010, 0.0046, 0.0080])
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            print('iou_anchors', iou_anchors)
            anchor_indices = iou_anchors.argsort(descending = True, dim = 0)
            # box [0.639, 0.5675675675675675, 0.718, 0.8408408408408409, 6.0]
            x, y, width, height, class_label = box
            # every box will only have one target in every scale
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale # 0,1,2
                anchor_on_scale = anchor_idx // (self.num_anchors // self.num_anchors_per_scale)# 0,1,2
                S = self.S[scale_idx]
                i, j = int(S*y), int(S*x) # x =0.5,S=13 --> int(6.5) = 6(which cell)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell,y_cell= S*x - j,S*y - i # both are between [0,1]
                    width_cell,height_cell= width *S, height *S# S = 13， width = 0.5 6.5
                    box_coordinate = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinate
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                #  which means this scale already has an anchor on other cell and this cell not taken 
                # and this anchor's iou is smaller  because it's after sorting
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

            return image, tuple(targets)

if __name__ == "__main__":
    dataset = YOLODataset(
    csv_file = './Datasets/PASCAL_VOC/1examples.csv',
    img_dir='./Datasets/PASCAL_VOC/images',
    label_dir='./Datasets/PASCAL_VOC/labels',
    anchors=[[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],]
    )
    anchors=[[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],]
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        # plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
        plot_image(x[0].to("cpu"), boxes)

