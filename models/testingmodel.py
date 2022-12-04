# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# !pip install -q efficientnet_pytorch
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
from sklearn.metrics import roc_auc_score

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

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.dense_output = nn.Linear(1280, num_classes)


    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)


def macro_multilabel_auc(label, pred):
    aucs = []
    for i in range(len(CFG.FOLDER_NAMES)):
        aucs.append(roc_auc)
    print(np.round(aucs, 4))
    return np.mean(aucs)


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

# output = torch.argmax(outputs, dim=1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4
model = Net(num_classes)
def train_one_epoch(num_epochs, train_loader, model, device, optimizer, criterion):
    summary_loss = AverageMeter()
    model.train()
    i = 0    
    epoch_loss = 0
    loader = tqdm(train_loader, total=len(train_loader))
    for imgs, labels in loader:
        i += 1
#         imgs = [torch.stack(img).to(device, dtype=torch.float) for img in imgs]
        imgs = torch.stack(imgs).to(device, dtype=torch.float)
        labels = torch.tensor(labels).to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(imgs)

        

        loss = criterion(outputs, labels)
        loss.backward()
        summary_loss.update(loss.detach().item(), CFG.TRAIN_BS)
        optimizer.step() 
        

    print(f'Loss after epoch {num_epochs+1} = ',summary_loss.avg)
    return summary_loss.avg
    
def val_one_epoch(num_epochs, val_loader, model, device, optimizer, criterion):
    model.eval()
    epoch_loss = 0
    loader = tqdm(val_loader, total=len(val_loader))
    label_val, preds = [], []
#     eval_scores = EvalMeter()
    running_loss = 0
    for imgs, labels in loader:
        imgs = torch.stack(imgs).to(device, dtype=torch.float)
        labels = torch.tensor(labels).to(device, dtype=torch.long)
        outputs = model(imgs)
        # 
        
        running_loss += loss.item()
        val_preds = torch.argmax(outputs, 1).detach().cpu().numpy()
        val_y = labels.detach().cpu().numpy()
        
        label_val += [val_y]
        preds += [val_preds]
        
        valid_pbar_desc = f"loss: {loss.item():.4f}"
        loader.set_description(desc=valid_pbar_desc)
#         
    final_loss_val = running_loss/len(loader)
#     print('Validation loss = ', summary_losses.avg)
    return final_loss_val
                
#                 prediction = model([img.to(device)])[0]

def engine():
# to train on gpu if selected.
#     num_classes = 4
    # get the model using our helper function
#     model = get_model_instance_segmentation(num_classes)
    num_epochs = CFG.EPOCHS
    # move model to the right device
    model.to(device)
    


    # parameters construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    precision_max = 0.940
    loss_min=99999
    loss = []
    # insert train_one_epoch and val_one_epoch
    for epochs in range(num_epochs):
        print(f"============Epoch: {epochs+1}============")
        losses_train = train_one_epoch(epochs, data_loader, model, device, optimizer, criterion)
        losses_val = val_one_epoch(epochs, val_data_loader, model, device, optimizer, criterion)
        loss.append(losses_val)
        
        if losses_val < loss_min:
            PATH = f'./FasterRCNN_epoch_bestLosses.pt'
            torch.save({
                'epoch':epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses_val
            }, PATH)
            print(f'Best loss found in {epochs+1}, with loss of {losses_val}... saving model to {PATH}')
            loss_min = losses_val
                  
#         if precision_val > precision_max:
#             PATH = f'./FasterRCNN_epoch_bestPrecision.pt'
#             torch.save({
#                 'epoch':epochs,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': losses_val
#             }, PATH)
#             print(f'Best precision found in {epochs+1}, with precision of {precision_val}... saving model to {PATH}')
#             precision_max = precision_val
        
#         if losses_val < loss_min and precision_val < precision_max:
#             PATH = f'./FasterRCNN_epoch_bestModel.pt'
#             torch.save({
#                 'epoch':epochs,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': losses_val
#             }, PATH)
#             print(f'Best overall score found in {epochs+1}, with loss of {losses_val}, and precision is {precision_val}... saving model to {PATH}')
#             precision_max = precision_val
#             loss_min = losses_val
        
        torch.cuda.empty_cache()

engine()

#see images
model.eval()
for i in range(10):
    img, target = dataset[i]
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        prediction = model(img)
    img = img.squeeze(0)
    img = img.permute(1,2,0)
    img = img.cpu().numpy()
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    print(target)




