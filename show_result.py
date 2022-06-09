from utils.dataset import TeethDataset

teeth_rgb = "teeth_rgb.json"

import torch
import torchvision
from torch import nn

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import albumentations as A
import albumentations.augmentations.transforms as T
from albumentations.pytorch import ToTensorV2

import pandas as pd
from models.maskrcnn import get_model_instance_segmentation

def tensor2image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor.permute(1, 2, 0), dtype=np.uint8)
  return Image.fromarray(tensor)


if __name__ == "__main__":
  
  model = get_model_instance_segmentation(2)
  
  param_dict = torch.load("best_model.pth")
  model.load_state_dict(param_dict)
  model.eval()

  # Test 1 image
  image_path = "Anterior-teeth-frontal-view.png"

  # Test all image
  files = pd.read_csv("_file.csv")

  # bounding box threshold
  bb_thre = 0.3

  fnt = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=15)
  for index in range(len(files)):
    file_name = files.iloc[:,0][index]
    sample_number = files.iloc[:,1][index]

    # image = Image.open(f"data/sample/{sample_number}/{file_name}")
    image = Image.open(image_path)

    images = torch.as_tensor(np.array(image)[...,:3]/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    output = model(images)[0]

    image = tensor2image(images[0])
    img = image.copy()
    draw = ImageDraw.Draw(img)
    scores = output["scores"]
    boxes = output["boxes"]
    for box, score in zip(boxes, scores):
      if score.item() > bb_thre:
        box = [b.item() for b in box]
        x1, y1, x2 ,y2 = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        color = tuple(np.random.choice(range(256), size=3))
        draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], fill=color, width=5)
        draw.text((x1,y1), f"{score.item():.4f}", font=fnt)
    # img.save(f"result/det_{file_name[:file_name.find('.')]}.png")
    img.save(f"det_{image_path}")
    
    masks = output["masks"]
    img = np.array(image.copy(), dtype=np.uint8)
    for mask,score in zip(masks, scores):
      if score.item() > bb_thre:
        mask = mask.detach().squeeze().numpy()
        color = list(np.random.choice(range(256), size=3))
        img[np.where(mask>0.8)] = color
    img = Image.fromarray(img)
    # img.show()
    # img.save(f"result/seg_{file_name[:file_name.find('.')]}.png")
    img.save(f"seg_{image_path}")
    exit()
