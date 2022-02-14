import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from imagedataset import ImageDataset, ImageDataLoader
from imagenet import ImageNet
import numpy as np
import os
import sys

class ImagePytorch:
    def __init__(self, net:ImageNet):
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        self.criteron = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device={}".format(self.device))
        
    
    def train(self, dataloader_train:ImageDataLoader, dataloader_test:ImageDataLoader, epoch_num):
        self.net.to(self.device)
        
        for epoch in range(epoch_num+1):
            for is_train in [True, False]:
                
                if is_train == True:
                    if epoch == 0:
                        continue
                    
                    dataloader = dataloader_train
                    self.net.train()
                else:
                    dataloader = dataloader_test
                    self.net.eval()
                
                epoch_loss_sum = 0.0
                epoch_correct_num = 0
                
                for inputs, labels in tqdm(dataloader()):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(is_train):
                        outputs = self.net(inputs)
                        loss = self.criteron(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        _, answers = torch.max(labels, 1)
                        
                        if is_train == True:
                            loss.backward()
                            self.optimizer.step()
                        
                        epoch_loss_sum += loss.item() * inputs.size(0)
                        epoch_correct_num += torch.sum(preds == answers)
                        
                epoch_loss = epoch_loss_sum / len(dataloader.dataset)
                epoch_acc = epoch_correct_num.double() / len(dataloader.dataset)
            
                print("epoch_num={}, train={}, loss={:.4f}, acc={:.4f}".format(epoch, is_train, epoch_loss, epoch_acc))
                
    def predict(self, dataloader:ImageDataLoader):
        all_outputs = None
        all_preds = None
        
        for inputs, _ in tqdm(dataloader()):
            inputs = inputs.to(self.device)
            
            with torch.set_grad_enabled(False):
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                
            outputs = outputs.to(self.device).detach().numpy().copy()
            preds = preds.to(self.device).detach().numpy().copy()
            all_outputs = np.concatenate([all_outputs, outputs])
            all_preds = np.concatenate([all_preds, preds])
                
        return all_preds, all_outputs