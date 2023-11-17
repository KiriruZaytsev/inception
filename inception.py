import torch
import torch.nn as nn
import torch.cuda

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super().__init__()

        self.branch1 = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(in_channels, out_1x1, kernel_size=(1, 1)),
                                     nn.BatchNorm2d(out_1x1))
        
        self.branch2 = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(in_channels, red_3x3, kernel_size=(1, 1)),
                                     nn.BatchNorm2d(red_3x3),
                                     nn.ReLU(),
                                     nn.Conv2d(red_3x3, out_3x3, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(out_3x3))
        
        self.branch3 = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(in_channels, red_5x5, kernel_size=(1, 1)),
                                     nn.BatchNorm2d(red_5x5),
                                     nn.ReLU(),
                                     nn.Conv2d(red_5x5, out_5x5, kernel_size=(5,5), padding=2),
                                     nn.BatchNorm2d(out_5x5))
        
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels, out_1x1pool, kernel_size=1),
                                     nn.BatchNorm2d(out_1x1pool))
        
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
    

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(in_channels=in_channels, out_channels=64, 
                                             kernel_size=(7, 7), stride=2, padding=3),
                                   nn.BatchNorm2d(64))
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=192, 
                                             kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(192))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)

        self.lin = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool3(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.maxpool4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.lin(x)
        return x


