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

import os
import os.path
from os import path
from sklearn.metrics import roc_auc_score

class CFG:
    BATCH_SIZE = 64
    VAL_BS = 8
    TRAIN_BS = 4
    EPOCHS = 1
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
sample_size = 9000
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
train_df = pd.DataFrame({'file_path': train_fn, 'Label': train_labels}, columns=['file_path', 'Label'])
train_df['Label'] = train_df['Label'].astype(int)
val_df = pd.DataFrame({'file_path': val_fn, 'Label': val_labels}, columns=['file_path', 'Label'])
val_df['Label'] = val_df['Label'].astype(int)

print(train_df)
train_df.Label.hist()



# class LandDataset(torch.utils.data.Dataset):

#     def __init__(self, images_dir, width, height, transforms=None):
#         self.transforms = transforms
#         self.images_dir = images_dir
#         self.height = height
#         self.width = width
        
#         # sorting the images for consistency
#         # To get images, the extension of the filename is checked to be jpg
#         self.imgs = [image for image in sorted(os.listdir(images_dir))]       
#         # classes: 0 index is reserved for background
#         self.classes = CFG.LABELS
#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):
#         img_name = self.imgs[idx]
#         image_id = torch.tensor([idx])
#         target = [int(idx/len(image_id)>1)]
#         file_path = os.path.join(self.images_dir, img_name)
#         img = cv2.imread(file_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
#         img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
#         img_res /= 255.0
        
#         labels = []
#         for i in range(len(self.classes)):
#             print(img_name)
#             if self.classes[i] in img_name:
#                 labels.append(i)
#         # convert everything into a torch.Tensor
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#         target = {}
#         target["labels"] = labels
        
#         target["image_id"] = image_id

#         if self.transforms:
#             transforms = self.transforms(image = img_res) 
#             img_res = transforms['image']
#         return img_res, target
class LandDataset(torch.utils.data.Dataset):
    def __init__(self, df, width, height, transforms=None):
        self.df = df
        self.width = width
        self.height = height
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['file_path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (self.width, self.height), cv2.INTER_AREA)
        img /= 255.0
        
        label = self.df.iloc[idx]['Label']
        if self.transforms:
            img = self.transforms(image=img)['image']
#         img = torch.tensor(img, dtype=torch.float)
#         label = torch.tensor(label, dtype=torch.long)
        return img, label


# check dataset
dataset = LandDataset(train_df, 500, 500)
print('length of dataset = ', len(dataset), '\n')

# getting the image and target for a test index.  Feel free to change the index.
img, target = dataset[15]
print('Image shape = ', img.shape, '\n','Target - ', target)
print(label)

# Function to visualize bounding boxes in the image

def plot_img(img, target):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    print(target)
    plt.axis('off')
    plt.show()
    
# plotting the image with bboxes. Feel free to change the index
img, target = dataset[7000]
plot_img(img, target)

# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):
    
    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                            A.RandomBrightnessContrast(p=0.2),
                            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ])
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ])

def collate_fn(batch):
    return tuple(zip(*batch))

# use our dataset and defined transformations
dataset = LandDataset(train_df, 500, 500, transforms= get_transform(train=True))
dataset_test = LandDataset(val_df, 500, 500, transforms= get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
# test_split = 0.9
# tsize = int(len(dataset)*test_split)
#test
# dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=10, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

val_data_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=10, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

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

        
iou_thresholds = [0.5]


class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
        
class EvalMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.image_precision = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, gt_boxes, pred_boxes, n=1):       
        """ pred_boxes : need to be sorted."""
        
        self.image_precision = calculate_image_precision(pred_boxes,
                                                         gt_boxes,
                                                         thresholds=iou_thresholds,
                                                         form='pascal_voc')
        self.count += n
        self.sum += self.image_precision * n
        self.avg = self.sum / self.count

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
        
#         print(f'Iteration: {i}/{len(loader)}, Loss: {losses}')
#         epoch_loss += losses.item()
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
        loss = criterion(outputs, labels)
        
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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4
model = Net(num_classes)

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
            PATH = f'./Efnetb0BestLosses.pt'
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

#Run engine
engine()

model.eval()
# predict images   
for imgs, labels in test_loader:
    imgs = torch.stack(imgs).to(device, dtype=torch.float)
    labels = torch.tensor(labels).to(device, dtype=torch.long)
    outputs = model(imgs)
    preds = torch.argmax(outputs, 1).detach().cpu().numpy()
    print(preds)
