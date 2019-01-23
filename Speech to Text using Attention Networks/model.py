# In[1]:


import argparse
import csv
import os
import sys
import psutil
import numpy as np
import torch
from torch.nn import init
torch.cuda.empty_cache()
import gc
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence,pack_sequence
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


voice_train,text_train = np.load('./all/train.npy', encoding = 'bytes'), np.load('./all/train_transcripts.npy')
voice_dev, text_dev = np.load('./all/dev.npy', encoding = 'bytes'), np.load('./all/dev_transcripts.npy')


# In[4]:

#
# subset = 10000
# voice_train,text_train = voice_train[:subset],text_train[:subset]

def read_corpus(text):
    lines = []
    for pos, line in enumerate(text):
        if len(line) > 0:
            line = line.astype(str)
            lines.append(list(str(line).lower()))
    for i in range(len(lines)):
        lines[i] = list(filter(lambda a: a != "'", lines[i]))
        lines[i] = list(filter(lambda a: a != "[", lines[i]))
        lines[i] = list(filter(lambda a: a != "]", lines[i]))
        lines[i] = list(filter(lambda a: a != "\n", lines[i]))
        lines[i] = list(filter(lambda a: a != '"', lines[i]))
        lines[i] = list(filter(lambda a: a != '-', lines[i]))
        lines[i] = list(filter(lambda a: a != '.', lines[i]))
        lines[i] = list(filter(lambda a: a != '_', lines[i]))
    return lines

def get_charmap(corpus):
    chars = list(set(np.concatenate(corpus)))
    chars.sort()
    charmap = {'sos':0, 'eos' :1}
    charmap.update({c: i+2 for i, c in enumerate(chars)})
    #del charmap['\n']
    return chars, charmap

def map_corpus1(corpus, charmap):
    out = []
    for i in range(len(corpus)):
        out.append(np.array([charmap[c] for c in corpus[i]], dtype=np.int32))
    return out

# In[32]:
class TextSpeechDataset(Dataset):
    def __init__(self,text,speech):
        self.speech = [torch.from_numpy(x).float() for x in speech]
        self.text = [torch.from_numpy(x).long() for x in text]
    def __getitem__(self,i):
        return self.speech[i], self.text[i]
    def __len__(self):
        return len(self.text)


# In[9]:


def speech_text_collatefn(seq_list):
    n = len(seq_list)
    utt_lens = torch.IntTensor(n).zero_()
    trans_lens = torch.IntTensor(n).zero_()
    for i, (ut, trans) in enumerate(seq_list):
        utt_lens[i] = ut.size(0)
        trans_lens[i] = trans.size(0) + 1

    max_utt_lens = torch.max(utt_lens)
    max_trans_lens = torch.max(trans_lens)

    utt = torch.FloatTensor(n, max_utt_lens, 40).zero_()
    trans_input = torch.LongTensor(n, max_trans_lens).zero_()
    trans_target = torch.LongTensor(n, max_trans_lens).zero_()

    for i, (ut, trans) in enumerate(seq_list):
        utt[i, :ut.size(0), :] = ut
        trans_input[i, 1:trans.size(0) + 1] = trans
        trans_target[i, :trans.size(0)] = trans

    return utt, utt_lens, trans_input, trans_lens, trans_target

text = np.concatenate((text_train, text_dev), axis = 0)
charset = read_corpus(text)
train_chars = read_corpus(text_train)
dev_chars = read_corpus(text_dev)
_, charmap = get_charmap(charset)
trainchars = map_corpus1(train_chars, charmap)
devchars = map_corpus1(dev_chars, charmap)

for i in range(len(trainchars)):
    trainchars[i] = np.insert(trainchars[i], 0, charmap['sos'], axis=None)
    trainchars[i] = np.append(trainchars[i], charmap['eos'], axis=None)


for i in range(len(devchars)):
    devchars[i] = np.insert(devchars[i], 0, charmap['sos'], axis=None)
    devchars[i] = np.append(devchars[i], charmap['eos'], axis=None)

total_trainchars = 0
total_devchars = 0
for i in range(len(trainchars)):
    total_trainchars += len(trainchars[i])

for i in range(len(devchars)):
    total_devchars += len(devchars[i])

train_dataset = TextSpeechDataset(trainchars, voice_train)
train_loader = DataLoader(train_dataset, collate_fn=speech_text_collatefn, shuffle = True, batch_size=128)
dev_dataset = TextSpeechDataset(devchars, voice_dev)
dev_loader = DataLoader(dev_dataset, collate_fn=speech_text_collatefn, batch_size=64)


class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.00):
        super(pBLSTMLayer, self).__init__()
        self.BLSTM = nn.LSTM(input_feature_dim,hidden_dim,1, bidirectional=True,
                                   dropout=dropout_rate,batch_first=True)

    def forward(self,input_x):
        pad, lengths = pad_packed_sequence(input_x, batch_first=True)
        if pad.size(1) % 2 > 0:
            pad = pad[:, :-1, :]
        pad = pad.contiguous().view(pad.size(0),pad.size(1)//2,pad.size(2)*2)
        plen = lengths//2
        input_x = pack_padded_sequence(pad, plen, batch_first=True)
        output,hidden = self.BLSTM(input_x)
        return output,hidden


# In[34]:


class Encoder(nn.Module):
    def __init__(self, *args):
        super(Encoder, self).__init__()
        self.rnns=nn.ModuleList()
        self.rnns.append(nn.LSTM(40, 256, bidirectional = True, batch_first = True))
        self.rnns.append(pBLSTMLayer(1024,256))
        self.rnns.append(pBLSTMLayer(1024,256))
        self.project_key = nn.Linear(512, 128)
        self.project_value = nn.Linear(512, 128)

    def forward(self, seq, lens):
        h = seq
        seq_lens, order = torch.sort(lens, 0, descending=True)
        dummy, backorder = torch.sort(order, 0)
        h = h[order]
        h = pack_padded_sequence(h, seq_lens, batch_first=True)
        h = h.to(DEVICE)
        for rnn in self.rnns:
            h, _ = rnn(h)
        h, seq_lens_out  = pad_packed_sequence(h, batch_first=True)
        h = h [backorder]
        seq_lens_out = seq_lens_out[backorder]
        keys = self.project_key(h)
        values = self.project_value(h)
        return keys, values , seq_lens_out

def create_mask(maxLen, seq_lens):
    mask = torch.zeros(len(seq_lens), maxLen)
    for i in range(len(mask)):
        mask[i][:seq_lens[i]] = 1
    return mask


def calculation_attn(keys, queries, mask):
    mask = mask.to(DEVICE)
    energy = torch.bmm(keys, queries.unsqueeze(2)).squeeze(2)
    softmax_energy = nn.functional.softmax(energy)
    softmax_energy_mask = softmax_energy * mask
    att = nn.functional.normalize(softmax_energy_mask, p = 1)
    return att

def calculate_cntxt(attention, values):
    a = attention.unsqueeze(1)
    b = torch.bmm(a, values)
    return b.squeeze(1)


# In[37]:


class LSTMCELL(nn.LSTMCell):
    def __init__(self, *args):
        super(LSTMCELL, self).__init__(*args)
        self.h_0 = nn.Parameter(torch.FloatTensor(1, 512).zero_())
        self.c_0 = nn.Parameter(torch.FloatTensor(1, 512).zero_())

    def init_state(self, n):
        return (self.h_0.expand(n, -1).contiguous(), self.c_0.expand(n, -1).contiguous())


# In[38]:


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.rnns = nn.ModuleList()
        self.rnns.append(LSTMCELL(512 + 128, 512))
        self.rnns.append(LSTMCELL(512, 512))
        self.rnns.append(LSTMCELL(512, 512))
        self.project_query = nn.Linear(512, 128)
        self.project_char = nn.Sequential(nn.Linear(512 + 128, vocab_size))

    def single_pass(self, curr_char, keys, values, mask, ctxt, input_states):
        char_embed = self.embedding(curr_char)
        hidden = torch.cat((char_embed, ctxt), dim = 1)
        new_states = []
        for rnn, states in zip(self.rnns, input_states):
            hidden, newstate = rnn(hidden, states)
            new_states.append((hidden, newstate))
        query = self.project_query(hidden)
        att = calculation_attn(keys, query, mask)
        ctxt = calculate_cntxt(att, values)
        hidden = torch.cat((hidden, ctxt), 1)
        out = self.project_char(hidden)
        c, gen = torch.max(out, dim = 1)
        return out, gen, ctxt, att, new_states

    def forward(self, input_txt, input_txt_lengths, keys, values, seq_lens):
        n = input_txt.size(0)
        input_states = [rnn.init_state(n) for rnn in self.rnns]
        input_txt = input_txt.transpose(0,1)
        initial = input_states[-1][0]
        query = self.project_query(initial)
        mask = Variable(create_mask(values.size(1), seq_lens)).float()
        att = calculation_attn(keys, query, mask)
        cntxt = calculate_cntxt(att, values)

        logits = []
        generateds = []
        attentions = []
        for i in range(input_txt.size(1)):
            input_t = input_txt[i]
            logit, generated, cntxt, att, input_states = self.single_pass(curr_char = input_t.to(DEVICE), keys = keys,
                                                                          values = values, mask = mask, ctxt = cntxt,
                                                                          input_states = input_states)
            logits.append(logit)
            attentions.append(att)
            generateds.append(generated)
        return logits, attentions, generateds

# In[56]:


class AttentionModel(nn.Module):
    def __init__(self, vocab_size):
        super(AttentionModel, self).__init__()
        self.encoder = Encoder()
        self.decode = Decoder(vocab_size=vocab_size)

    def forward(self, seq, seq_lens, text, text_lengths):
        keys, values, lens = self.encoder(seq, seq_lens)
        logits, attentions, generated = self.decode(text, text_lengths, keys, values, lens)
        return logits, generated, text_lengths, attentions


# In[57]:


class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1):
        """
            Use this class to train your model
        """
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
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-5)
        #self.optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum = 0.9)

        self.criterion = nn.CrossEntropyLoss.to(DEVICE)

    def train(self):
        self.model.train() # set to training mode
        epoch_perp = 0
        epoch_loss = 0
        num_batches = 0
        for batch_num, (utt, utt_lens, trans_input, trans_lens, trans_target) in enumerate(self.loader):
            perp, loss = self.train_batch(batch_num, utt, utt_lens, trans_input, trans_lens, trans_target)
            epoch_perp += perp
            epoch_loss += loss
        epoch_perp = epoch_perp / total_trainchars
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d] LOSS: %.4f PERP: %.4f'
                      % (self.epochs, self.max_epochs, epoch_loss, np.exp(epoch_perp.cpu().detach().numpy())))
        #self.train_losses.append(epoch_loss)

    def train_batch(self, batch_num, utt, utt_lens, trans_input, trans_lens, trans_target):
        """
            TODO: Define code for training a single batch of inputs
        """
        utt = utt.to(DEVICE).float()
        utt_lens = utt_lens.to(DEVICE).int()
        trans_input = trans_input.to(DEVICE).long()
        trans_lens = trans_lens.to(DEVICE).int()
        trans_target = trans_target.to(DEVICE).long()
        logits, generated, text_lengths , attentions = self.model(utt, utt_lens, trans_input, trans_lens)
        logits = torch.stack(logits)
        maxlen = logits.size(0)
        mask = Variable(create_mask(maxlen, text_lengths.data), requires_grad = False).float()
        mask = mask.to(DEVICE)
        target = target.to(DEVICE)
        logits = logits.transpose(0,1)
        logits = logits.transpose(1,2)
        losses = self.criterion.forward(logits, target)
        loss = losses * mask
        perp = torch.sum(loss)
        loss = torch.sum(loss, dim = 1)
        loss = torch.mean(loss)
        attent = torch.stack(attentions)
        plt.matshow(attent[:,0,:].cpu().detach().numpy())
        plt.savefig('plot1.png', format='png')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return perp, loss.item()

    def test(self, dev_loader):
        # don't change these
        dev_loss = 0
        PERP = 0
        self.model.eval() # set to eval mode
        for batch_num, (utt, utt_lens, trans_input, trans_lens, trans_target) in enumerate(dev_loader):
            logits, generated, text_lengths, attentions = self.model(utt, utt_lens, trans_input, trans_lens)
            logits = torch.stack(logits)
            maxlen = logits.size(0)
            mask = Variable(create_mask(maxlen, text_lengths.data), requires_grad = False).float()
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)
            logits = logits.transpose(0,1)
            logits = logits.transpose(1,2)
            losses = self.criterion.forward(logits, target)
            loss = losses * mask
            perp = torch.sum(loss)
            loss = torch.sum(loss, dim = 1)
            loss = torch.mean(loss)
            dev_loss += loss.item()
            PERP += perp
            self.generated_logits_test.append(logits)
            self.generated_test.append(generated)
        perp_loss_dev = PERP/total_devchars
        dev_loss = dev_loss / (batch_num + 1)
        print('[DEV] LOSS: %.4f PERP: %.4f'
                      % (dev_loss, np.exp(perp_loss_dev.cpu().detach().numpy())))
        return perp_loss_dev

model = AttentionModel(len(charmap))
NUM_EPOCHS = 30
model.load_state_dict(torch.load("train_start4.pt")) #loading currently trained mode
trainer = LanguageModelTrainer(model=model, loader=train_loader, max_epochs=NUM_EPOCHS)
model = model.to(DEVICE)
best_nll = 1e30  # set to super large value at first
for epoch in range(NUM_EPOCHS):
    trainer.train()
    #torch.save(model.state_dict(), "train_start5.pt")
    print("Starting validation")
    with torch.no_grad():
        nll = trainer.test(dev_loader)
        if nll < best_nll:
            best_nll = nll
            print("Saving model, predictions and generated output for epoch " +
                  str(epoch)+" with NLL: " + str(best_nll))
            torch.save(model.state_dict(), "check.pt")
