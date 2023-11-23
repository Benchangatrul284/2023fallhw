import torch
import torch.nn as nn
from thop import profile



class CIFAR_part1(nn.Module):
    def __init__(self, num_classes = 10):
        super(CIFAR_part1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 64, num_classes),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        out = self.softmax(out)
        return out
    


    
class CIFAR_part2(nn.Module):
    def __init__(self, num_classes = 10):
        super(CIFAR_part2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout(p = 0.5)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 64, num_classes),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=1)
        # self.reset_parameters()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        out = self.softmax(out)
        return out

class CIFAR_part3(nn.Module):
    def __init__(self, num_classes: int):
        super(CIFAR_part3, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256,256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,8, 5, 1, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2),
       )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256,256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,8, 5, 1, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
       )
        
        self.linear1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(p = 0.2),
            nn.ReLU(True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(p = 0.2),
            nn.ReLU(True)
        )
        self.out = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        output = self.out(x)
        return output
    


def convblock(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, 1, 2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(p = 0.2)
    )

class block(nn.Module):
    def __init__(self,in_channels):
        super(block, self).__init__()
        self.in_channels = in_channels
        self.conv1 = convblock(in_channels,8)
        self.conv2 = convblock(8,64)
        self.conv3 = convblock(64,256)
        self.conv4 = convblock(256,256)
        self.conv5 = convblock(256,64)
        self.conv6 = convblock(64,8)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = x3 + x4
        x5 = self.conv5(x4)
        x5 = x2 + x5
        x6 = self.conv6(x5)
        x6 = x1 +x6
        return x6





class CIFAR_part4(nn.Module):
    def __init__(self, num_classes: int):
        super(CIFAR_part4, self).__init__()
        self.num_classes = num_classes
        self.block1 = block(3)
        self.block2 = block(8)
        self.linear1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.out = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = nn.MaxPool2d(2)(x)
        x = self.block2(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    # calulate flops and params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR_part2(num_classes=10).to(device)
    input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))