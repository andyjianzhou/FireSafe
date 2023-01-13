import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.autonotebook import tqdm
import random
from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms, datasets, models
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import shutil
# numba
import numba
from numba import jit
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# xml library for parsing xml files
from xml.etree import ElementTree as et
import cv2
from sklearn.metrics import roc_auc_score
import torchvision.transforms as T
import os
import os.path
from os import path
import PIL

class LandDataset(torch.utils.data.Dataset):

  def __init__(self, imgs, width, height, transforms=None, real_time = True):
      self.transforms = transforms
      self.imgs = imgs
      self.height = height
      self.width = width
      
      # sorting the images for consistency
      # To get images, the extension of the filename is checked to be jpg
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    #convert PIL image to opencv image
    # img = self.imgs
    # img = np.array(img)
    img = convert_from_image_to_cv2(self.imgs)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (self.width, self.height)).astype(np.float32)
    img /= 255.0
    if self.transforms:
        transforms = self.transforms(image = img) 
        img_res = transforms['image']
    return img_res



class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # 1280 is the number of neurons in last layer. is diff for diff. architecture
        self.dense_output = nn.Linear(1280, num_classes)

    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)
def get_transform(train):
    
    if train:
        return A.Compose([
                            #A.HorizontalFlip(0.5),
                            #A.RandomBrightnessContrast(p=0.2),
                            #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ],)
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ],)

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
def threshold(output):
    print("Thresholding")
    output = torch.sigmoid(output) # sigmoid function to convert to probability values to be between 0 and 1
    probability = output
    print(output)
    output = output.cpu().detach().numpy()
    output = np.where(output > 0.999, 1, 0)
    return output, probability
def torch_to_pil(img):
    # img is a torch tensor, convert to PIL image
    print(type(img))
    return transforms.ToPILImage()(img).convert('RGB')


def get_predictions(image, width, height, model_path):
    LABELS = ['Mali', 'Ethiopia', 'Malawi', 'Nigeria']
    model_path =  model_path #insert path to model
    model = Net(len(LABELS))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    print("Model loaded")

    model.eval()
    model.to(device)

    imgs = LandDataset(image, width, height, transforms = get_transform(False))
    img = imgs[0]
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)[0]
    # output = output.cpu().detach().numpy()
    output, probability = threshold(output)
    print("Output: ", output) # the array contents the probability of each class
    img = torch_to_pil(img[0]) #convert to PIL image
    return img, output, probability
    
def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)