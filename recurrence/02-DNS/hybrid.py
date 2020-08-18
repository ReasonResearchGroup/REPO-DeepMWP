import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from retrieval import Retrieval
from RNN import DNS
class Hybrid(nn.Module):
    def __init__(self, seq2seq, retrieval, cuda_use=False):
        super(Hybrid,self).__init__()
        self.seq2seq = seq2seq
        self.retrieval = retrieval
        self.cuda_use = cuda_use
    def forward(self, input_text,input_variable, input_lengths=None, target_variable=None, retrieval_use=True, \
                rule_use=True,teacher_forcing_ratio=0,use_rule=True):
        simi_list=[]
        similarity=0
        if retrieval_use:
            similarity, gen_temp,text = self.retrieval.js(input_text)
        if similarity > self.retrieval.theta:
            return 1,gen_temp
        else:
            outputs_list, decoder_hidden, sequence_symbols_list = self.seq2seq(input_variable,
                                                                                input_lengths,
                                                                                target_variable,
                                                                                teacher_forcing_ratio,
                                                                                self.cuda_use)
            return  2,sequence_symbols_list
            