import random

import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .encode import *
from .decode import *

class Mathen(nn.Module):
    def __init__(self, encoder, decoder, data_loader=None, cuda_use=False, decoder_function=F.log_softmax):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = data_loader
        self.cuda_use = cuda_use
        self.decoder_function = decoder_function
        
            
    def forward(self, input_variable, input_lengths=None, target_variable=None, \
                teacher_forcing_ratio=1, mode=0, cuda_use=False):
        if self.cuda_use:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        encoder_hidden = encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])

        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decoder_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              cuda_use = cuda_use,
                              class_list=self.dataloader.decode_classes_list,
                              vocab_dict=self.dataloader.vocab_2_ind)
        return result
    def _cat_directions(self, h):
        if self.encoder.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
class DNS(nn.Module):
    def __init__(self, encoder, decoder, data_loader, cuda_use=False, decoder_function=F.log_softmax):
        super(DNS,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = data_loader
        self.cuda_use=cuda_use
        self.decoder_function = decoder_function
    def forward(self, input_variable, input_lengths=None, target_variable=None,teacher_forcing_ratio=1,use_rule=True):
        if self.cuda_use:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
        
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        if self.encoder.cell_name == self.decoder.cell_name:
            pass
        elif self.encoder.cell_name=='gru' and self.decoder.cell_name=='lstm':
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif self.encoder.cell_name == 'lstm' and self.decoder.cell_name == 'gru':
            encoder_hidden = encoder_hidden[0]
        
        result = self.decoder(inputs=target_variable,
                            encoder_hidden=encoder_hidden,
                            encoder_outputs=encoder_outputs,
                            function=self.decoder_function,
                            cuda_use = self.cuda_use,
                            class_list=self.dataloader.decode_classes_list,
                            vocab_dict=self.dataloader.vocab_2_ind,
                            class_dict=self.dataloader.decode_classes_2_ind,
                            vocab_list=self.dataloader.vocab_list,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            use_rule=use_rule)
        return result
