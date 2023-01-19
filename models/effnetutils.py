import numpy as np  # linear algebra
from torchvision import transforms
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# xml library for parsing xml files
import cv2


class LandDataset(torch.utils.data.Dataset):
    """
    Args:
        imgs (list): list of image paths
        width (int): width of the image
        height (int): height of the image
        transforms (albumentations.Compose): data transformations

    Returns:
        torch.Tensor: transformed image
    """

    def __init__(self, imgs, width, height, transforms=None, real_time=True):
        self.transforms = transforms
        self.imgs = imgs
        self.height = height
        self.width = width

    # Needs to be implemented so that len(dataset) returns the size of the dataset, required by Pytorch
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = convert_from_image_to_cv2(self.imgs)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height)).astype(np.float32)
        img /= 255.0
        if self.transforms:
            transforms = self.transforms(image=img)
            img_res = transforms['image']
        return img_res


class Net(nn.Module):
    """
    EfficientNet model for classification

    Args:
        num_classes (int): number of classes
    """

    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # 1280 is the number of neurons in last layer. is diff for diff. architecture
        self.dense_output = nn.Linear(1280, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, 3, H, W)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, num_classes)
        """
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)


def get_transform(train):
    if train:
        return A.Compose([
            # A.HorizontalFlip(0.5),
            # A.RandomBrightnessContrast(p=0.2),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            # ToTensorV2 converts image to pytorch tensor without div by 255
            ToTensorV2(p=1.0)
        ],)
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ],)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    """
    Args:
        img (Image): PIL image

    Returns:
        np.ndarray: OpenCV image
    """
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def threshold(output):
    print("Thresholding")

    # sigmoid function to convert to probability values to be between 0 and 1
    output = torch.sigmoid(output)
    probability = output
    output = output.cpu().detach().numpy()

    # threshold the output to 0 or 1, 0.999 is the threshold
    output = np.where(output > 0.999, 1, 0)

    return output, probability


def torch_to_pil(img):
    # img is a torch tensor, convert to PIL image
    print(type(img))
    return transforms.ToPILImage()(img).convert('RGB')

# Main prediction function
def get_predictions(image, width, height, model_path):
    LABELS = ['Mali', 'Ethiopia', 'Malawi', 'Nigeria']
      
    model = Net(len(LABELS))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    print("Model loaded")

    model.eval()
    model.to(device)

    # use main LandDataset class to preprocess the image for training
    imgs = LandDataset(image, width, height, transforms=get_transform(False))
    # get the first image
    img = imgs[0]

    # add a batch dimension, in this case it adds a dimension of 1 
    # because there is only one image
    img = img.unsqueeze(0)
    img = img.to(device) 
    output = model(img)[0]

    output, probability = threshold(output)
    # the array contents the probability of each class
    print("Output: ", output)
    print("Probability: ", probability)
    img = torch_to_pil(img[0])  # convert to PIL image
    return img, output, probability
