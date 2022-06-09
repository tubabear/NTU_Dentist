import torch
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

class TeethDataset(torch.utils.data.Dataset):
    """return image, target(mask)"""
    def __init__(self, root, csv_file,teeth_rgb, transforms=None, device='cpu'):
        self.root = root
        self.files = pd.read_csv(csv_file)
        self.transforms = transforms

        with open(teeth_rgb, 'r') as f:
            color_dict = json.loads(f.read())[0]
            # self.obj_ids = np.array([np.array(color_dict[k][::-1], dtype=np.uint8) for k in color_dict])
            self.obj_ids = torch.as_tensor(np.array([np.array(color_dict[k][::-1], dtype=np.uint8) for k in color_dict]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """load images and masks"""
        file = self.files.iloc[index, 0]
        sample_id = self.files.iloc[index, 1]
        img_path = os.path.join(self.root, "sample", str(sample_id), file)
        mask_path = os.path.join(self.root, "mask", str(sample_id), f"{file[:file.find('.')]}.png")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # change image, mask into numpy format
        img = np.array(img, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        if self.transforms is not None: # apply data augmentation
            # transformed = self.transforms(image=img, mask=mask, bboxes=boxes)
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
            # boxes = np.array(transformed["bboxes"])
        else: 
            mask = mask.as_tensor(mask, dtype=torch.uint8)

        # split the color-encoded mask into a set of binary masks
        masks = (mask == self.obj_ids[:, None, None]).all(-1)

        # get bounding box coordinates for each mask
        boxes = []
        empty_masks = []
        for i in range(masks.shape[0]):
            try: # if conatins this color
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            except ValueError: # empty mask
                empty_masks.append(i)

        # don't keep empty mask
        masks = np.delete(masks.numpy(), empty_masks, axis=0)

        boxes = np.array(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # only one class
        labels = np.array(np.ones(len(boxes), dtype=np.int64))

        target = {}
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target["boxes"] = boxes
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target["labels"] = labels
        image_id = torch.tensor([index])
        target["image_id"] = image_id
        area = torch.as_tensor(area, dtype=torch.float32)
        target["area"] = area
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        target["masks"] = masks
        img = torch.as_tensor(img/255, dtype=torch.float32)

        return img, target
"""
file_root = "../data/"
csv_file = "../file_.csv"
teeth_rgb = "../teeth_rgb.json"

import albumentations as A
import albumentations.augmentations.transforms as T
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
                        A.LongestMaxSize(1024),
                        T.PadIfNeeded(1024,1024, border_mode=0,),
                        A.Rotate(limit=180, border_mode=0, p=1),
                        ToTensorV2(),
                        ],
            )

dataset = TeethDataset(file_root, csv_file, teeth_rgb, transforms=transforms)
for i in range(len(dataset)):
    img, target_dict = dataset[i]
    img = img.numpy()
    img = np.moveaxis(img, 0, -1)
    img = np.array(img*255, dtype=np.uint8)

    masks = target_dict["masks"]
    for mask in masks:
        mask = mask.numpy()
        color = list(np.random.choice(range(256), size=3))
        img[np.where(mask)] = color

    img = Image.fromarray(img)

    boxes = target_dict["boxes"]
    areas = target_dict["area"]
    for box, area in zip(boxes, areas):
        x1, y1, x2, y2 = box
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())
        color = tuple(np.random.choice(range(256), size=3))
        draw = ImageDraw.Draw(img)
        draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], width=3, fill=color)
        draw.text((x1,y1), f"{int(area.item())}")

    img.show()
"""