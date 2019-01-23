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

class TextSpeechDataset(Dataset):
    def __init__(self,text,speech):
        self.speech = [torch.from_numpy(x).float() for x in speech]
        self.text = [torch.from_numpy(x).long() for x in text]
    def __getitem__(self,i):
        return self.speech[i], self.text[i]
    def __len__(self):
        return len(self.text)

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

model = AttentionModel(len(charmap))
model.load_state_dict(torch.load("train_start4.pt")) #loading currently trained mode

out = []
dum = ""
sample = 1
out.append(sample)
test_text = torch.LongTensor([[sample]])
logits, attentions, generateds = model.decode(test_text, test_text_len, keys, values, seq_lens_out)
sample = torch.argmax(logits[0]).item()
test_text = torch.LongTensor([[sample]])
while(sample != 1):
    dum = dum + chars[sample - 2]
    print(sample)
    out.append(sample)
    logits, attentions, generateds = model.decode(test_text, test_text_len, keys, values, seq_lens_out)
    sample = torch.argmax(logits[0]).item()
    test_text = torch.LongTensor([[sample]])
