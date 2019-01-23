
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation


# In[2]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


# In[3]:


# load all that we need

dataset = np.load('../dataset/wiki.train.npy')
fixtures_pred = np.load('../fixtures/dev_fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/dev_fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/test_fixtures/prediction.npz')  # test
fixtures_gen_test = np.load('../fixtures/test_fixtures/generation.npy')  # test
vocab = np.load('../dataset/vocab.npy')


# In[4]:


fixtures_pred['out'].shape


# In[6]:


dataset[1].shape


# In[5]:


# data loader

class LanguageModelDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        
        self.data = dataset
        self.batch_size = batch_size
        self.cuda = False

    def __iter__(self):
        # concatenate your articles and build into batches
        shuffled = np.concatenate(np.random.permutation(self.data)) 
        count = len(shuffled)
        num_batches = (count - 1)//self.batch_size
        number = num_batches * self.batch_size
        x = shuffled[0:number].reshape(self.batch_size, num_batches).T
        y = shuffled[1:number+1].reshape(self.batch_size, num_batches).T
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.cuda:
            x = x.cuda()
            y = y.cuda()
        
        start = 0
        while start < num_batches:
            if (np.random.uniform() < 0.95):
                L = round(np.random.normal(70,5))
            else:
                L = round(np.random.normal(35,5))
            x_out = x[start:start + L]
            y_out = y[start:start + L]
            start = start + L
            yield x_out,y_out 


# In[6]:


# model

class LanguageModel(nn.Module):
    def __init__(self, charcount):
        super(LanguageModel, self).__init__()
        self.vocab_size = charcount
        self.embedding = nn.Embedding(self.vocab_size,400)
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(400,1150))
        self.rnns.append(nn.LSTM(1150,1150))
        self.rnns.append(nn.LSTM(1150,400))
        self.linear = nn.Linear(400, self.vocab_size)
        
    def forward(self, seq_batch):
        batch_size = seq_batch.size(1)
        embed = self.embedding(seq_batch)
        hiddens = []
        for layers in self.rnns:
            embed, state = layers(embed)
            hiddens.append(state)
        output_lstm_flatten = embed.view(-1,400)
        outputs = self.linear(output_lstm_flatten)
        #print(outputs.size())
        return outputs.view(-1,batch_size,self.vocab_size)


# In[7]:


# model trainer

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-6)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)

    def train(self):
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(inputs, targets)
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        inputs = inputs.to(DEVICE).long()
        targets = targets.to(DEVICE).long()
        outputs = self.model(inputs)
        loss = self.criterion(outputs.view(-1,outputs.size(2)),targets.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        print(predictions.shape)
        self.predictions.append(predictions)
        nll = test_prediction(predictions, fixtures_pred['out'])
        
        generated_logits = TestLanguageModel.generation(fixtures_gen, 10, self.model) # predictions for 20 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model) # predictions for 20 words

        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   NLL: %.4f'
                      % (self.epochs, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-test-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


# In[8]:


class TestLanguageModel:
    def prediction(inp, model):
        inp = torch.from_numpy(inp.T)
        inp = inp.to(DEVICE).long()
        h = model(inp)
        h = h[-1,:,:]
        return h.cpu().data.numpy()
        #return h.detach().numpy()
        
    def generation(inp, forward, model):
        inp = torch.from_numpy(inp.T)
        inp = inp.to(DEVICE).long()
        h = model(inp)
        word = h[-1,:,:]
        word = torch.max(word, dim = 1)[1]
        words = torch.unsqueeze(word,1)   
        h = torch.max(h,dim = 2)[1]
        for i in range(forward - 1):
                output = model(h)
                word = output[-1,:,:]
                word = torch.max(word, dim = 1)[1]
                word = torch.unsqueeze(word,1)
                h = torch.max(output,dim = 2)[1]
                words = torch.cat((words, word), dim=1)
        return words


# In[10]:


#hyperparameters here

NUM_EPOCHS = 7
BATCH_SIZE = 64


# In[11]:


run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)


# In[12]:


model = LanguageModel(len(vocab))
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)


# In[13]:


model = model.to(DEVICE)


# In[ ]:


best_nll = 1e30  # set to super large value at first
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch " + 
              str(epoch)+" with NLL: " + str(best_nll))
        trainer.save()

