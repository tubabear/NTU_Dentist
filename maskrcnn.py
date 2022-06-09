from utils.dataset import TeethDataset

file_root = "data/"
csv_file = "file_.csv"
teeth_rgb = "teeth_rgb.json"

import torch
import torchvision
from utils.trainer import train
from models.maskrcnn import get_model_instance_segmentation
from torch import nn

import albumentations as A 
import albumentations.augmentations.transforms as T
from albumentations.pytorch import ToTensorV2

def get_transform(train):
  transforms = []
  if train:
    transforms.append(A.LongestMaxSize(1024))
    transforms.append(T.PadIfNeeded(1024,1024, border_mode=0))
    transforms.append(A.Rotate(limit=180, border_mode=0, p=1))
  transforms.append(ToTensorV2())  
  return A.Compose(transforms)

def main():
  # train on the GPU or on the CPU, if a GPU is not available
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  # our dataset has two classes only - background and tooth
  num_classes = 2
  # use our dataset and defined transformations
  dataset = TeethDataset(file_root, csv_file, teeth_rgb, get_transform(train=True))

  # split the dataset in train and test set
  indices = torch.randperm(len(dataset)).tolist()
  # dataset = torch.utils.data.Subset(dataset, indices[:-50])
  # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

  # define training and validation data loaders
  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=lambda x:tuple(zip(*x)))
  print(f"{len(dataset)} training images.")

  # data_loader_test = torch.utils.data.DataLoader(
  #     dataset_test, batch_size=1, shuffle=False, num_workers=4,
  #     collate_fn=utils.collate_fn)

  # get the model using our helper function
  model = get_model_instance_segmentation(num_classes)

  # move model to the right device
  model.to(device)

  # construct an optimizer
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005,
                              momentum=0.9, weight_decay=0.0005)
  epoch = 10
  train(model, optimizer, data_loader, device, epoch)

if __name__ == "__main__":
  main()
