import torch
import torch.nn as nn
import os
#import your model here
from model import CIFAR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

def show_results(imgs, label, pred):
    num_imgs = len(imgs)
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    correct = 0
    for i in range(num_imgs):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(imgs[i].permute(1, 2, 0))
        l = label[i].item() 
        p = pred[i].item()
        if l == p:
            correct += 1
        axs[row, col].set_title('label: {}, pred: {}'.format(l, p))
    
    plt.tight_layout()
    plt.savefig('results.png')
    print('accuracy: {} out of {}'.format(correct, num_imgs))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([    
                                transforms.Resize((64,64)),
                                transforms.ToTensor(),
                            ])
    test_data = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=False,
        transform=transform,
    )
    test_dl = DataLoader(test_data, batch_size=16, shuffle=True,num_workers=4)
    
    model = CIFAR(num_classes=10).to(device)
    model.load_state_dict(torch.load('checkpoint/model.pth')['model'],strict=False)
    model.eval()
    for img,label in test_dl:
        img = img.to(device)
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        # print(pred)
        show_results(img.cpu(), label.cpu(),pred.cpu())
        break