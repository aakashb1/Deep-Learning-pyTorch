#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os
import torch.utils.data as Data
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from hw2.all_cnn import all_cnn_module
from hw2.all_cnn_base import all_cnn_basemodule
from hw2.utils import *
#from hw2.model_res3 import *

# In[2]:
torch.cuda.empty_cache()
model = all_cnn_module(2413)
#model.load_state_dict(torch.load('model_14epochs.ckpt'))
gpu=True
if gpu:
    model = model.cuda() # the network parameters also need to be on the gpu !
    print("Using GPU")
else:
    print("Using CPU")

path = "./hw2p2_A/"
x_train, y_train , n_1 = train_load(path, range(1,6))
#print(y_train[0:8])

def padding(utterance):
    length = 9000
    ref = len(utterance)
    if (ref <= length):
        x = np.pad(utterance,[(0,length - ref),(0,0)],'wrap')
    else:
        start_ind = np.random.randint(0, ref - length)
        end_ind = start_ind + length
        x = utterance[start_ind:end_ind]
    return x

#x_train = np.asarray(list(map(padding,x_train)))
# In[4]:

def model_train(gpu, batch_size, x_train, y_train):
    #model = all_cnn_module(n_1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum = 0.9, weight_decay = 0.001, nesterov=True)
    num_epochs = 2
    #total_step = len(load_train)

    #if gpu:
    #    model = model.cuda() # the network parameters also need to be on the gpu !
    #    print("Using GPU")
    #else:
    #    print("Using CPU")

    for epoch in range(num_epochs):
        s = np.arange(len(x_train))
        np.random.shuffle(s)
        x_train = x_train[s]
        y_train = y_train[s]
        for i in range(0, len(x_train), batch_size):
            torch.cuda.empty_cache()
            images = np.asarray(list(map(padding, x_train[i:i+batch_size])))
            #images = x_train[i:i+batch_size]
            #print(images.shape)
            images = images.astype('float32')
            labels = torch.from_numpy(y_train[i:i+batch_size])
            images = torch.from_numpy(images[:, np.newaxis])
            # Forward pass
            if gpu:
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images, True)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 75 == 0:
                print ('Epoch [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, loss.item()))
        
    return model


# In[5]:


mod = model_train(gpu, 32, x_train, y_train)


# In[ ]:

torch.save(mod.state_dict(), "./" + "model_14epochs.ckpt")


