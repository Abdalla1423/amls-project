import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset

K = 16
IN_CHANNELS = 3

# Custom Dataset class to load images and labels from our cleaned parquet file, with data augmentation for training
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std_max=0.02):
        self.mean = mean
        self.std_max = std_max

    def __call__(self, tensor):
        std = random.uniform(0.0, self.std_max)
        if std == 0:
            return tensor
        return tensor + torch.randn(tensor.size()) * std + self.mean
    
# Custom Dataset class to load images and labels from our cleaned parquet file, also augment training dataset
class AugmentedImageDataset(Dataset):
    def __init__(self, images, labels, is_training):
        self.images = images
        self.labels = labels
        self.is_training = is_training

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
            ], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.1),
            T.RandomApply([AddGaussianNoise(mean=0.0, std_max=0.015)], p=0.1),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx])).float()

        # Apply augmentations
        if self.is_training:
            img = (img * self.std) + self.mean
            img = torch.clamp(img, 0.0, 1.0)

            img = self.train_transforms(img)
            img = torch.clamp(img, 0.0, 1.0)
            
        img = self.normalize(img)
        return img, torch.tensor(self.labels[idx]).long()

# Custom Dataset class to load images and labels from our cleaned parquet file
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx])).float()
        return img, torch.tensor(self.labels[idx]).long()
    
# Provided CNN model
class Given_CNN(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=2, k=K):
        super(Given_CNN, self).__init__()

        # Block 1:
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(k)
        self.pool1 = nn.MaxPool2d(2)

        self.block1 = nn.Sequential(self.conv1, self.bnorm1, nn.ReLU(), self.pool1)

        # Block 2:
        self.conv2 = nn.Conv2d(in_channels=k, out_channels=2*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(2*k)
        self.pool2 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(self.conv2, self.bnorm2, nn.ReLU(), self.pool2)

        # Block 3:
        self.conv3 = nn.Conv2d(in_channels=2*k, out_channels=4*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(4*k)
        self.pool3 = nn.MaxPool2d(2)

        # Block 4:
        self.conv4 = nn.Conv2d(in_channels=4*k, out_channels=4*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(4*k)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.block3 = nn.Sequential(self.conv3, self.bnorm3, nn.ReLU(),
                                    self.conv4, self.bnorm4, nn.ReLU(),
                                    self.global_pool)

        # Classifier:
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(4*k, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.bnorm1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bnorm2(out)
        out += skip
        return nn.functional.relu(out)
    
# Custom CNN Model - A smaller and more efficient version of AlexNet, with global average pooling
class Custom_CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Custom_CNN, self).__init__()

        # Block 1:
        # Input 64x64x3 -> Output 32x32x16
        # Change the layers Alexnet, where it removes most of the AI artifacts from the images
        # Alexnet uses out channel 96, kernel_size 11 and stride 4.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(16)
        # Input 32x32x16 -> Output 16x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1 = nn.Sequential(self.conv1, self.bnorm1, nn.ReLU(), self.pool1)

        # Block 2:
        # Input 16x16x16 -> Output 8x8x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(32)
        # Input 8x8x32 -> Output 8x8x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = nn.Sequential(self.conv2, self.bnorm2, nn.ReLU(), self.pool2)

        # Block 3:
        # Input 8x8x32 -> Output 8x8x64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(64)
        self.res_block = ResidualBlock(64)

        # Input 8x8x64 -> Output 2x2x64
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.block3 = nn.Sequential(self.conv3, self.bnorm3, nn.ReLU(),
                                    self.res_block,
                                    self.global_pool)

        # Block 4:
        self.classifier = nn.Sequential(
            # Shape : 1x256
            nn.Flatten(),
            # Shape : 1x64
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # Shape : 1x2
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

class ImageDatasetRAM(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class ImageDatasetDISK(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx])).float()
        return img, torch.tensor(self.labels[idx]).long()
    
