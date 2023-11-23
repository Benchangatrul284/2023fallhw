import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
from torchvision import transforms



class resnet(nn.Module):
    def __init__(self, num_classes: int):
        super(resnet, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet50(weights ='IMAGENET1K_V1')
        self.linear = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Dropout(p = 0.5),
            nn.ReLU(True)
        )
        self.out = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    # calulate flops and params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet(num_classes=10).to(device)
    input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))