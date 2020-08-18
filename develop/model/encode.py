import random

import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MathenEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size=100, hidden_size=128, \
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, \
                 variable_lengths=True):
        super(MathenEncoder, self).__init__()
        """
        parameter : vocab_size : size of vocabulary
        parameter : emb_size   : size of embedding vector
        parameter : hidden_size: size of hidden layer
        parameter : dropout_p  : rate of rnn's dropout
        parameter : n_layers   : number of hidden layer
        parameter : input_dropout_p : rate of dropout after embedding
        parameter : bidirectional   : true or false to control rnn's direction
        """
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers,
                          batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
    def forward(self, input_var, input_lengths=None):
        """
        parameter : input_var     : input sequence variables
        parameter : input_lengths : lenths of input sequences
        ---------------------------------
        output    : output and hidden of rnn
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
class DNSEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, input_dropout, dropout, n_layer, cell_name='gru'):
        super(DNSencoder, self).__init__()
        """
        parameter : vocab_size : size of vocabulary
        parameter : emb_size   : size of embedding vector
        parameter : hidden_size: size of hidden layer
        parameter : dropout_p  : rate of rnn's dropout
        parameter : n_layer    : number of hidden layer
        parameter : cell_name  : type of rnn, expect gru or lstm
        parameter : input_dropout_p : rate of dropout after embedding
        """
        self.embedding = nn.Embedding(vocab_size, emb_size,)
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.cell_name = cell_name
        if self.cell_name == 'gru':
            self.rnn = nn.GRU(emb_size, hidden_size, n_layer, batch_first=True)
        elif self.cell_name == 'lstm':
            self.rnn=nn.LSTM(emb_size, hidden_size, n_layer, batch_first=True)
    
    def forward(self, input_var, input_lengths=None):
        """
        parameter : input_var     : input sequence variables
        parameter : input_lengths : lenths of input sequences 
        ---------------------------------
        output    : output and hidden of rnn
        """
        embedded = self.embedding(input_var)
        #print(embedded.size())
        embedded = self.input_dropout(embedded)
        #print(embedded.size())
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
