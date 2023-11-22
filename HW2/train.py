import argparse
import os
# import your model here
from model import CIFAR
from resnet import resnet
import math

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import plot_accuracy_curve, plot_loss_curve, plot_lr_curve, plot_class_accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', help='path to latest checkpoint')
parser.add_argument('--export', default='model.pth', help='path to save checkpoint')
parser.add_argument('--epoch', default=50, help='number of epochs to train')
parser.add_argument('--batch_size', default=4096, help='batch size')
parser.add_argument('--lr', default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, help='weight decay')
args = parser.parse_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def adjust_learning_rate(epoch, T_max=args.epoch, eta_min=args.lr*0.5, lr_init=args.lr):
    lr = eta_min + (lr_init - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    if epoch >= T_max:
        lr = eta_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    history = []
    train_loss = 0
    best_accuracy = 0
    start_epoch = 1
    #loading pretrained models
    if args.resume:
        if os.path.isfile(args.resume):
            print("===> loading models '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            history = checkpoint['history']
            print("checkpoint loaded: epoch = {}, accuracy = {}".format(start_epoch, best_accuracy))
        else:
            print("===> no models found at '{}'".format(args.resume))

    for epoch in range(start_epoch,args.epoch + 1):
        adjust_learning_rate(epoch)
        result = {'train_loss': [], 'valid_loss': [], 'lrs': [], 'accuracy': []}
        print('Epoch: {}'.format(epoch))
        print('learning rate: {:.6f}'.format(optimizer.param_groups[0]['lr']))
        model.train()
        for (img,label) in tqdm(train_dl):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dl) # average loss per batch
        
        model.eval()
        with torch.no_grad():
            # compute validation loss 
            valid_loss = 0
            for (img,label) in tqdm(valid_dl):
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                loss = criterion(output, label)
                valid_loss += loss.item()
            valid_loss = valid_loss / len(valid_dl) # average loss per batch
            # compute test accuracy
            correct = 0
            class_correct = list(0. for i in range(num_classes))
            class_total = list(0. for i in range(num_classes))

            for (img,label) in tqdm(test_dl):
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                pred = output.argmax(dim=1, keepdim=True) #returns the index of the maximum value
                correct += pred.eq(label.view_as(pred)).sum().item()

                _, predicted = torch.max(output, 1)
                c = (predicted == label).squeeze()
                for i in range(len(label)):
                    label_i = label[i]
                    class_correct[label_i] += c[i].item()
                    class_total[label_i] += 1

            for i in range(num_classes):
                print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
            class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(num_classes)]

            plot_class_accuracy(class_accuracies,classes,epoch)
                
           
                
                
        accuracy = correct/len(test_data)
        result['train_loss'].append(train_loss)
        result['valid_loss'].append(valid_loss)
        result['accuracy'].append(accuracy)
        result['lrs'].append(optimizer.param_groups[0]['lr'])
        print('Train Loss: {:.4f}'.format(train_loss))
        print('Val Loss: {:.4f}'.format(valid_loss))
        print('Test Accuracy: {:.4f}'.format(accuracy))
        history.append(result)

        if accuracy > best_accuracy:
            model_folder = "checkpoint"
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            best_accuracy = accuracy
            model_out_path = os.path.join(model_folder, args.export)
            state = {"epoch": epoch,
                    "model": model.state_dict(),
                    "best_accuracy": best_accuracy,
                    "history": history}
            torch.save(state, model_out_path)
            print("===> Checkpoint saved to {}".format(model_out_path))

        plot_loss_curve(history)
        plot_accuracy_curve(history)
        plot_lr_curve(history)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans1 = transforms.Compose([    
                                transforms.Resize((32,32)),
                                transforms.ToTensor(),
                            ])
    trans2 = transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                            ])
    
    train_data1 = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=True,
        transform=trans1,
        download=True,
    )

    train_data2 = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=True,
        transform=trans2,
        download=True,
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=False,
        transform=trans1,
        download=True
    )
    # split the training and validation dataset
    train_data = torch.utils.data.ConcatDataset([train_data1,train_data2])

    train_ds, valid_ds = torch.utils.data.random_split(train_data, [90000, 10000])
    # dataloader
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True,num_workers=4)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,num_workers=4)
    # model
    # model = CIFAR(num_classes=10).to(device)
    model = resnet(num_classes=10).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    num_classes = 10
    classes = ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
    train()
    