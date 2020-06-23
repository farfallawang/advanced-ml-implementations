#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:49:59 2020

@author: flora
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets
import random
import matplotlib.pyplot as plt

random.seed(123)

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden1, hidden2): 
        super().__init__()
        self.encode1 = nn.Linear(input_size, hidden1)
        self.encode2 = nn.Linear(hidden1, hidden2)
        self.decode1 = nn.Linear(hidden2, hidden1)
        self.decode2 = nn.Linear(hidden1, input_size)

    def encode(self, x):
        x = torch.relu(self.encode1(x))
        x = torch.relu(self.encode2(x))
        return x
    
    def decode(self, x):
        x = torch.relu(self.decode1(x))
        x = torch.sigmoid(self.decode2(x))
        return x
    
    def forward(self, x):
        encoded_x = self.encode(x)
        decoded_x = self.decode(encoded_x)
        return decoded_x
  
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
model = AutoEncoder(input_size, hidden1, hidden2)
opt = optim.Adam(model.parameters(), lr = 0.001)   
train_loss_lst = []
epochs = 10

for epoch in range(epochs):
    loss = 0.0
    print('Epoch', epoch, '.... \n')
    
    for train_X, _ in train_dl:
        original_img = train_X.view(len(train_X), -1) #batch_size * 784
        noise = torch.randn([len(train_X), input_size])
        new_img = model(original_img + noise)
        loss_per_pixel = nn.BCELoss(reduction = 'sum')(new_img, original_img) #summation over both dimension
        
        opt.zero_grad()
        loss_per_pixel.backward()
        opt.step()
        
        loss += loss_per_pixel.item()
        
    train_loss_lst.append(loss / (len(trainset) * 784)) 

# Save model
torch.save(model, 'hw5_dAE.pth')


''' Plot train loss '''
fig = plt.figure()
plt.plot(range(epochs), train_loss_lst, color='blue')
plt.xlabel('Number of epochs')
plt.ylabel('Train loss')
fig

''' Plot test image '''
columns = 5
rows = 2
fig = plt.figure(figsize=(columns, rows))
for i in range(1, columns + 1):
    original_test = testset[i][0].view(1, -1) 
    noise = torch.randn(size = original_test.shape)
    noisy_test = original_test + noise
    
    # --------------------
    #  Plot noisy test
    # --------------------
    fig.add_subplot(rows, columns, i)
    plt.imshow(noisy_test.detach().numpy().reshape(28,28) , cmap = "gray")
    
    # --------------------
    #  Plot denoised test
    # --------------------
    denoised_test = model(noisy_test)
    fig.add_subplot(rows, columns, i + 5)
    plt.imshow(denoised_test.detach().numpy().reshape(28,28) , cmap = "gray")
plt.show() 
