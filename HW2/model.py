import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
from torchvision import transforms


class CIFAR(nn.Module):
    def __init__(self, num_classes: int):
        super(CIFAR, self).__init__()
        self.num_classes = num_classes
        self.activation = nn.Softmax(dim = 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 4, 5, 1, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.out = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return self.activation(output)
    

if __name__ == '__main__':
    # calulate flops and params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR(num_classes=10).to(device)
    input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))