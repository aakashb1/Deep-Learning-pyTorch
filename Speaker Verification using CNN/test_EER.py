import numpy as np
import torch
import torch.nn as nn
import os
from hw2.utils import *
from hw2.all_cnn import all_cnn_module
from sklearn.metrics.pairwise import cosine_similarity

path = "./hw2p2_A/"
dev_idx, dev_data, test_data = test_load(path)
#test_idx, test_data, test_test_data = test_load(path)
mod = all_cnn_module(1397)
mod.load_state_dict(torch.load('model.ckpt'))

def padding(utterance):
    length = 5000
    ref = len(utterance)
    if (ref <= length):
        x = np.pad(utterance,[(0,length - ref),(0,0)],'wrap')
    else:
        start_ind = np.random.randint(0, ref - length)
        end_ind = start_ind + length
        x = utterance[start_ind:end_ind]
    return x

dev_data = np.asarray(list(map(padding,dev_data)))
test_data = np.asarray(list(map(padding,test_data)))


def model_val(mod, gpu, batch_size):
    scores=[]
    #mod.load_state_dict(torch.load('model.ckpt'))

    if gpu:
        mod = mod.cuda() # the network parameters also need to be on the gpu !
        print("Using GPU")
    else:
        print("Using CPU")

    with torch.no_grad():
        dev_embedding=[]
        test_embedding=[]
        print('Doing dev data')
        j = 1
        for i in range(0, len(dev_data), batch_size):
            images = dev_data[i:i+batch_size].astype('float32')
            #images = np.array(list(map(padding,dev_data[i:i+batch_size])))
            #images = images.astype('float32')
            images = torch.from_numpy(images[:, np.newaxis])
            if gpu:
                images = images.cuda()
            dev_embedding.append(mod(images,False))
            #print(dev_embedding[0].shape)
            if j % 50 == 0:
                 print('Dev Data', +i)
            j = j + 1

        print('Doing test_data')
        j = 1

        for i in range(0, len(test_data), batch_size):
            images = test_data[i:i+batch_size].astype('float32')
            images = torch.from_numpy(images[:, np.newaxis])
            if gpu:
                images = images.cuda()
            test_embedding.append(mod(images,False))
            #print(test_embedding[i].shape)
            if j % 40 == 0:
                 print('test data', +i)
            j = j + 1

        dev_embedding = torch.cat(dev_embedding).cpu().data.numpy().astype('float')
        test_embedding = torch.cat(test_embedding).cpu().data.numpy().astype('float')

        j = 1
        for dev_index, test_index in dev_idx:
            dev = dev_embedding[dev_index]
            test = test_embedding[test_index]
            scores.append(cosine_similarity(dev.reshape(1,-1), test.reshape(1,-1)))
            if j % 1000 == 0:
                 print('Computing Similarities', j)
            j = j + 1
        return np.squeeze(scores)

scores = model_val(mod, True, batch_size = 16)
print(scores.shape)
np.save('scores.npy', scores)

#eer, threshold = EER(dev_labels,scores)

#print(eer)
#print(threshold)
