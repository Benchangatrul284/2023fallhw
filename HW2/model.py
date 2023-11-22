import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
from torchvision import transforms


class CIFAR(nn.Module):
    def __init__(self, num_classes: int):
        super(CIFAR, self).__init__()
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
            nn.Dropout(p = 0.5),
            nn.ReLU(True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(p = 0.5),
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
    
    def reset_parameters():
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    # calulate flops and params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR(num_classes=10).to(device)
    input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))