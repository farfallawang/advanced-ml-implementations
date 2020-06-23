#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:08:30 2020

@author: flora
"""
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from torchvision.models import resnet50
from torch.autograd import Variable
random.seed(123)

# ------------------------------------
#  Load json file and process image
# ------------------------------------
def json_lookup(idx):
    return df[str(idx)][1]

with open('imagenet_class_index.json') as f:
    df = json.load(f)

preprocess = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor()
#                                 transforms.Normalize((0.485, 0.456, 0.406), (.229, 0.224, 0.225))
                                 ])
img = Image.open('Elephant2.jpg')
original_img = preprocess(img)[None,:,:,:]  #torch.Size([1, 3, 224, 330])

# ------------------------------------------------------
#  Load pretrained model and fix pretrained model params 
# ------------------------------------------------------
model = resnet50(pretrained=True)
model.eval()

for param in model.parameters():
    param.requires_grad = False    

# ----------------------------------
#  Predict on original example 
# ----------------------------------
pred_vector = model(original_img)  # torch.Size([1, 1000])
pred_idx = torch.argmax(pred_vector).item()
print('This image is predicted as:', json_lookup(pred_idx))

tmp = original_img.detach().numpy()[0]
transposed_img = np.transpose(tmp, [1,2,0])
plt.imshow(transposed_img)  

# ------------------------------
#  Adversarial training
# ------------------------------
global epochs, lr_rate, epsilon, lam1, lam2
epochs = 10
lr_rate = 0.1
epsilon = 0.1 #limit for noise 
lam1 = 10
lam2 = 0.001 


def generate_adversarial_noise(target_label):
    noise = torch.clamp(torch.randn(size = original_img.shape), -epsilon, epsilon)
    noise = Variable(noise, requires_grad = True)
    for epoch in range(epochs):
        print('Epoch', epoch, '.... \n')
        new_img = original_img + noise  
        new_img_pred = model(new_img) # torch.Size([1, 1000])
        print('Predicted label: ', json_lookup(torch.argmax(new_img_pred).item()))
        
        X_reconstruct_loss = torch.sum((new_img - original_img) **2)
        y_reconstruct_loss = nn.CrossEntropyLoss()(new_img_pred, torch.LongTensor([target_label]))
        loss = lam1 * y_reconstruct_loss + lam2 * X_reconstruct_loss 
        
        grad = torch.autograd.grad(loss, noise, create_graph=True)[0]
        noise = noise - lr_rate * grad  
        
    return noise

def adversarial_pred(target_label):
    noise = generate_adversarial_noise(target_label)
            
    new_img = original_img + noise
    pred_vector = model(new_img)  
    pred_idx = torch.argmax(pred_vector).item()
    
    transposed_final_img = np.transpose(new_img.detach().numpy()[0], [1,2,0])
    plt.imshow(transposed_final_img) 
    plt.title('Predicted label: ' + json_lookup(pred_idx))

# ------------------------------------------
#  Predict on adversarial example part 1
# ------------------------------------------
mud_turtle = 35
adversarial_pred(mud_turtle)

# ------------------------------------------
#  Predict on adversarial example part 2
# ------------------------------------------
bullet_train = 466
adversarial_pred(bullet_train)

