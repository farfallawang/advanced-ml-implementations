#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:18:36 2020

@author: flora
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets
import random
import matplotlib.pyplot as plt

random.seed(123)

class FeedForwardNN(nn.Module):
    
    def __init__(self, input_size, hidden1, hidden2, hidden3, last_activation): 
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.last_activation = last_activation

    def forward(self, x):
        x = F.leaky_relu_(self.fc1(x), 0.2)
        x = F.leaky_relu_(self.fc2(x), 0.2)
        x = self.last_activation(self.fc3(x))
        return x
  
''' Process data '''    
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)) #transforms.Normalize((0.5,), (0.5,))
                              ])  #
trainset = datasets.MNIST(root='./data', train = True, download=False, transform = transform)
testset = datasets.MNIST(root='./data', train = False, download=False, transform = transform)

train_bs = 100 
test_bs = 1000
train_dl = DataLoader(trainset, batch_size = train_bs)
test_dl = DataLoader(testset, batch_size = test_bs)

sample_batch = iter(train_dl).next()
input_size = sample_batch[0].shape[2] * sample_batch[0].shape[2] # 28 * 28

''' Training '''
generator = FeedForwardNN(128, 256, 512, input_size, torch.tanh)
discriminator = FeedForwardNN(input_size, 512, 256, 1, torch.sigmoid)

gen_opt = optim.Adam(generator.parameters(), lr = 0.001) #0003  
dis_opt = optim.Adam(discriminator.parameters(), lr = 0.001) #0003 

discriminator_loss_lst = []
generator_loss_lst = []
epochs = 50
columns = 4
rows = 4

for epoch in range(epochs):
    discriminator_loss = 0.0
    generator_loss = 0.0
    print('Epoch', epoch, '.... \n')
    batch = 0
    for train_X, _ in train_dl:
        
        true_data = train_X.view(train_X.shape[0], -1)
        input_noise = torch.randn(torch.Size((len(train_X), 128))) #the number of fake data need to equal batch size
        fake_data = generator(input_noise) #batch_size * 784
         
        # -----------------
        #  Train Discriminator
        # -----------------
        dis_opt.zero_grad()  
           
        true_pred = discriminator(true_data)
        fake_pred = discriminator(fake_data)
        ones = torch.ones([len(train_X), 1])
        zeros = torch.zeros([len(train_X), 1])
        dis_loss = nn.BCELoss(reduction = 'sum')(torch.cat((true_pred, fake_pred), 0), torch.cat((ones, zeros), 0))
        
        dis_loss.backward(retain_graph=True)
        dis_opt.step()
        
        discriminator_loss += dis_loss.item()
    
        # -----------------
        #  Train Generator
        # -----------------
        gen_opt.zero_grad()
        
        fake_pred = discriminator(fake_data)  
        gen_loss = nn.BCELoss(reduction = 'sum')(fake_pred, torch.ones([len(train_X), 1]))

        gen_loss.backward()
        gen_opt.step()
        
        generator_loss += gen_loss.item()
        
    if epoch % 10 == 0:
        fig = plt.figure(figsize=(columns, rows))
        for i in range(1, columns*rows +1):
            fake_img = fake_data[i].detach().numpy().reshape(28,28) 
            fig.add_subplot(rows, columns, i)
            plt.imshow(fake_img, cmap = "gray")
        plt.show() 
        
    discriminator_loss_lst.append(discriminator_loss/ len(trainset))
    generator_loss_lst.append(generator_loss/ len(trainset))
    
torch.save(generator, 'hw5_gan_gen.pth')
torch.save(discriminator, 'hw5_gan_dis.pth')

#generator = torch.load('hw5_gan_gen.pth')
#discriminator = torch.load('hw5_gan_dis.pth')


''' Plot figure '''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(epochs), discriminator_loss_lst, color='blue', label = 'discriminator')
ax.plot(range(epochs), generator_loss_lst, color='orange', label = 'generator')
leg = plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Generator & Discriminator loss')
plt.show