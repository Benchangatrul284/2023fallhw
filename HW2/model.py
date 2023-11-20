import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
from torchvision import transforms


class CIFAR(nn.Module):
    def __init__(self, num_classes: int):
        super(CIFAR, self).__init__()
        self.num_classes = num_classes

        # in[N, 3, 32, 32] => out[N, 16, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        # in[N, 16, 16, 16] => out[N, 32, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        # in[N, 32 * 8 * 8] => out[N, 128]
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 16 * 16, 256),
            nn.Dropout(0.5), # p=0.5
            nn.ReLU(True)
        )
        # in[N, 128] => out[N, 64]
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        # in[N, 64] => out[N, 10]
        self.out = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        return output
    

if __name__ == '__main__':
    # calulate flops and params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR(num_classes=10).to(device)
    input = torch.randn(1, 3, 64, 64).to(device)
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))