import torch
import torch.nn as nn

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

# Custom CNN Model - A smaller and more efficient version of AlexNet, with global average pooling and additional residual connection
class Custom_CNN(nn.Module): 
    def __init__(self, in_channels=3, num_classes=2): 
        super(Custom_CNN, self).__init__()
        # Block 1: # Input 64x64x3 -> Output 32x32x16 
        # Change the layers Alexnet, where it removes most of the AI artifacts from the images 
        # Alexnet uses out channel 96, kernel_size 11 and stride 4. 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bnorm1 = nn.BatchNorm2d(16) 
        # Input 32x32x16 -> Output 16x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.block1 = nn.Sequential(self.conv1, self.bnorm1, nn.ReLU(), self.pool1) 
        
        # Block 2: # Input 16x16x16 -> Output 8x8x32 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bnorm2 = nn.BatchNorm2d(32) 
        # Input 8x8x32 -> Output 8x8x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.block2 = nn.Sequential(self.conv2, self.bnorm2, nn.ReLU(), self.pool2) 
        
        # Block 3: # Input 8x8x32 -> Output 8x8x64 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bnorm3 = nn.BatchNorm2d(64) 
        self.res_block = ResidualBlock(64) 
        # Input 8x8x64 -> Output 2x2x64 
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2)) 
        self.block3 = nn.Sequential(self.conv3, self.bnorm3, nn.ReLU(), self.res_block, self.global_pool)
        
        # Block 4: 
        self.classifier = nn.Sequential( 
            # Shape : 1x256 
            nn.Flatten(), 
            # Shape : 1x64 
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(p=0.5),
            # Shape : 1x2 
            nn.Linear(64, num_classes) 
        ) 
    def forward(self, x): 
        x = self.block1(x) 
        x = self.block2(x) 
        x = self.block3(x) 
        x = self.classifier(x) 
        return x