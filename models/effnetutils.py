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
    img = convert_from_image_to_cv2(self.imgs)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
    img_res /= 255.0
    if self.transforms:
        transforms = self.transforms(image = img_res) 
        img_res = transforms['image']
    # print("after transforms: ", img_res.shape)
    return img_res

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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

def get_predictions(image, width, height, model_path):
    model_path =  '' #insert path to model
    num_classes = 4
    model = Net(num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])

    model.eval()
    model.to(device)

    imgs = LandDataset(image, width, height, transforms = None)
    imgs = imgs[0]
    imgs = imgs.to(device)
    output = model(imgs)
    output = output[0]

    return imgs, output

def torch_to_pil(img):
    print("Converting to PIL image")
    return transforms.ToPILImage()(img).convert('RGB')
def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)