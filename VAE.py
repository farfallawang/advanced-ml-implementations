#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:49:22 2020

@author: flora
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets
import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(123)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden1, hidden2): 
        super().__init__()
        self.encode1 = nn.Linear(input_size, hidden1)  #784 --> 400
        self.encode2 = nn.Linear(hidden1, hidden2 * 2) #400 --> 20
        self.decode1 = nn.Linear(hidden2, hidden1) #20 --> 400
        self.decode2 = nn.Linear(hidden1, input_size) #400 --> 784
        self.hidden2 = hidden2

    def encode(self, x):
        x = torch.relu(self.encode1(x))
        x = torch.relu(self.encode2(x))  # batch_size * 20
        mean, logvar = x[:, :self.hidden2], x[:, self.hidden2: ]
        return mean, logvar
    
    def decode(self, x):
        x = torch.relu(self.decode1(x))
        x = torch.sigmoid(self.decode2(x))
        return x
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        sqrt_det = torch.exp(1/2 * logvar)
        epsilon = torch.randn(size = mean.shape)
        encoded_x = mean + epsilon * sqrt_det #logvar 
        decoded_x = self.decode(encoded_x)
        return mean, logvar, decoded_x
  
''' Process data '''    
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)) 
                              ])  
trainset = datasets.MNIST(root='./data', train = True, download=False, transform = transform)
testset = datasets.MNIST(root='./data', train = False, download=False, transform = transform)

train_bs = 64
test_bs = 1000
train_dl = DataLoader(trainset, batch_size = train_bs)
test_dl = DataLoader(testset, batch_size = test_bs)

sample_batch = iter(train_dl).next()
input_size = sample_batch[0].shape[2] * sample_batch[0].shape[2] # 28 * 28

''' Training '''
hidden1 = 400
hidden2 = 20 
lr_rate = 0.001
model = VariationalAutoEncoder(input_size, hidden1, hidden2)
opt = optim.Adam(model.parameters(), lr = lr_rate)   

reconstruction_loss = nn.BCELoss(reduction = 'sum')
def KL(mean, logvar): 
    return 1/2 * torch.sum(1 + logvar - mean **2 - logvar.exp()) #mean, logvar: batch_size * 20
    

train_loss_lst = []
epochs = 10
for epoch in range(epochs):
    loss = 0.0
    print('Epoch', epoch, '.... \n')
    
    for train_X, _ in train_dl:
        original_x = train_X.view(len(train_X), -1) #batch_size * 784
        noise = torch.randn([len(train_X), input_size])
        mean, logvar, decoded_x = model.forward(original_x + noise)
        batch_loss = reconstruction_loss(decoded_x, original_x) - KL(mean, logvar)
        #print('Batch loss', batch_loss)
        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        
        loss += batch_loss.item()
        #print('Accumulated batch loss', loss)
        
    train_loss_lst.append(loss / (len(trainset)* 784)) 
    print('Loss', loss / (len(trainset)* 784))

# Save model
# torch.save(model, 'hw5_vae.pth')

# Load model
#model = torch.load('hw5_vae.pth')

''' Plot train loss '''
fig = plt.figure()
plt.plot(range(epochs), train_loss_lst, color='blue')
plt.xlabel('Number of epochs')
plt.ylabel('Negative log-likelihood')
fig

''' Plot test image '''
columns = 4
rows = 4
fig = plt.figure(figsize=(columns, rows))
for i in range(1, rows * columns + 1):
    noise = torch.randn([1, hidden2])
    decoded_x = model.decode(noise) # 1 * 784 
    fig.add_subplot(rows, columns, i)
    plt.imshow(decoded_x.detach().numpy().reshape(28,28) , cmap = "gray")
plt.show()     

