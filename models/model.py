# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install -q efficientnet_pytorch
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

import os
import os.path
from os import path

class CFG:
    BATCH_SIZE = 64
    VAL_BS = 8
    TRAIN_BS = 4
    EPOCHS = 100
    IMG_SIZE = 256
    NUM_WORKERS = 8
    SEED = 42069
    LR = 1e-4
    MIN_LR = 1e-6 # CosineAnnealingWarmRestarts
    WEIGHT_DECAY = 1e-6
    MOMENTUM = 0.9
    T_0 = EPOCHS # CosineAnnealingWarmRestarts
    MAX_NORM = 1000
    T_MAX = 5
    ITERS_TO_ACCUMULATE = 1
    #   BASE_OPTIMIZER = SGD #for Ranger
    OPTIMIZER = 'Adam' # Ranger, AdamW, AdamP, SGD
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    N_FOLDS = 5
    START_FOLDS = 0
#   LABELS = [_, 'Jun Mask Off','Nic Mask On','Nic Mask Off', 'Jun Mask On']
    DATA_PATH  = '../input/satellite-images-to-predict-povertyafrica/'
    MALI_PATH = DATA_PATH + 'Mali_archive' + 'images/'
    ETHIOPIA_PATH = DATA_PATH + 'ethiopia_archive' + 'images/'
    MALAWI_ARCHIVE = DATA_PATH + 'malawi_archive' + 'images/'
    NIGERIA_ARCHIVE = DATA_PATH + 'nigeria_archive' + 'images/'
    FOLDER_NAMES = [MALI_PATH, ETHIOPIA_PATH, MALAWI_ARCHIVE, NIGERIA_ARCHIVE] 
    
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(CFG.SEED)


train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Mali_archive/images/", "ethiopia_archive/images/", "malawi_archive/images/", "nigeria_archive/images/"] # label 1 2 3 4
sample_size = 10000
val_size = int(sample_size*0.25)
for label, folder in enumerate(folder_names):
    train_filenames = sorted(glob(f"{CFG.DATA_PATH}{folder}*.png")[:sample_size])
    np.random.shuffle(train_filenames)
    train_fn.extend(train_filenames[val_size:])
    train_labels.extend(np.zeros(len(train_filenames[val_size:],))+label)
    val_fn.extend(train_filenames[:val_size])
    val_labels.extend(np.zeros(len(train_filenames[:val_size],))+label)
    
assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"
train_df = pd.DataFrame({'ImageFileName': train_fn, 'Label': train_labels}, columns=['ImageFileName', 'Label'])
train_df['Label'] = train_df['Label'].astype(int)
val_df = pd.DataFrame({'ImageFileName': val_fn, 'Label': val_labels}, columns=['ImageFileName', 'Label'])
val_df['Label'] = val_df['Label'].astype(int)

print(train_df)
print(val_df)
class LandDataset(torch.utils.data.Dataset):
    def __init__(self, df, width, height, transforms=None):
        self.df = df
        self.width = width
        self.height = height
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        img_path = self.df.iloc[idx]['ImageFileName']
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (self.width, self.height), cv2.INTER_AREA)
        img /= 255.0
        
        label = self.df.iloc[idx]['Label']
        if self.transforms:
            img = self.transforms(image=img)['image']
        #convert img and label to tensor
        return img, label


# check dataset
dataset = LandDataset(train_df, 500, 500)
print('length of dataset = ', len(dataset), '\n')

# getting the image and target for a test index.  Feel free to change the index.
img, target = dataset[15]
print('Image shape = ', img.shape, '\n','Target - ', target)
print(label)


def train_one_epoch(num_epochs, train_loader, model, device, optimizer, criterion):
    loader = tqdm(train_loader, total=len(train_loader))
    summary_loss = AverageMeter()
    model.train()
    i = 0    
    epoch_loss = 0
    for imgs, labels in loader:
        i += 1
#         imgs = [torch.stack(img).to(device, dtype=torch.float) for img in imgs]
        imgs = torch.stack(imgs).to(device, dtype=torch.float)
        #label is a tuple and a ndarray
        #convert labels to tensor
        labels = torch.tensor(labels).to(device, dtype=torch.long)
        labels = torch.stack(labels)
    
        
        outputs = model(imgs)
        loss_dict = criterion(outputs, labels)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        summary_loss.update(loss.detach().item(), CFG.TRAIN_BS)
        optimizer.step() 
        
#         print(f'Iteration: {i}/{len(loader)}, Loss: {losses}')
#         epoch_loss += losses.item()
    print(f'Loss after epoch {num_epochs+1} = ',summary_loss.avg)
    return summary_loss.avg
    
def val_one_epoch(num_epochs, val_loader, model, device, optimizer, criterion):
    model.eval()
    epoch_loss = 0
    loader = tqdm(val_loader, total=len(val_loader))
#     summary_losses = AverageMeter()
    eval_scores = EvalMeter()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        
        for i, image in enumerate(imgs):

            scores = outputs[i]['scores'].detach().cpu().numpy()
            
            preds_sorted_idx = np.argsort(scores)[::1]
        
        #Uncomment for mode.train() evaluation if you want loss only
#         losses = sum(loss for loss in loss_val_dict.all())
#         summary_losses.update(summary_losses.item(), CFG.VAL_BS)
    print("Precision is: ", eval_scores.avg)
#     print('Validation loss = ', summary_losses.avg)
    return eval_scores.avg
                
#                 prediction = model([img.to(device)])[0]

