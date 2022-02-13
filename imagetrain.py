import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from imagedataset import ImageDataset, ImageDataLoader
from imagenet import ImageNet
import numpy as np
import os

def train_model(net:ImageNet, dataloader_train:ImageDataLoader, dataloader_test:ImageDataLoader, epoch_num):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criteron = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device={}".format(device))
    
    net.to(device)
    
    for epoch in range(epoch_num+1):
        for is_train in [True, False]:
            
            if is_train == True:
                if epoch == 0:
                    continue
                
                dataloader = dataloader_train
                net.train()
            else:
                dataloader = dataloader_test
                net.eval()
            
            epoch_loss_sum = 0.0
            epoch_correct_num = 0
            
            for inputs, labels in tqdm(dataloader()):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(is_train):
                    outputs = net(inputs)
                    loss = criteron(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    _, answers = torch.max(labels, 1)
                    
                    if is_train == True:
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss_sum += loss.item() * inputs.size(0)
                    epoch_correct_num += torch.sum(preds == answers)
                    
            epoch_loss = epoch_loss_sum / len(dataloader.dataset)
            epoch_acc = epoch_correct_num.double() / len(dataloader.dataset)
        
            print("epoch_num={}, train={}, loss={:.4f}, acc={:.4f}".format(epoch, is_train, epoch_loss, epoch_acc))

if __name__=="__main__":
    dataset_train = ImageDataset(True, 32)
    dataset_train.load_numpys(r"D:\git\pytorch_test\cifar10\cifar10_data\train")
    dataloader_train = ImageDataLoader(dataset_train, 50)
    
    dataset_test = ImageDataset(False, 32)
    dataset_test.load_numpys(r"D:\git\pytorch_test\cifar10\cifar10_data\test")
    dataloader_test = ImageDataLoader(dataset_test, 50)
        
    net = ImageNet(3, 32, 32, 10)
    
    data_path=r"D:\git\pytorch_test\testdata.dat"
    if os.path.isfile(data_path):
        net.load_weight(data_path)
        
    train_model(net, dataloader_train, dataloader_test, 2)
    
    net.save_weight(data_path)
    