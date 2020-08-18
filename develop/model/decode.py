import random

import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .Sublayer import *
class MathenDecoder(nn.Module):
    def __init__(self, vocab_size, class_size, emb_size=100, hidden_size=128, \
                n_layers=1, sos_id=1, eos_id=0, input_dropout_p=0, dropout_p=0, cuda_use=False):
        super(MathenDecoder, self).__init__()
        """
        parameter : vocab_size : size of vocabulary
        parameter : class_size : size of symbols
        parameter : emb_size   : size of embedding vector
        parameter : hidden_size: size of hidden layer
        parameter : dropout_p  : rate of rnn's dropout
        parameter : n_layers   : number of hidden layer
        parameter : input_dropout_p : rate of dropout after embedding
        parameter : sos_id     :
        parameter : eos_id     :
        parameter " cuda_use   : true or false to control using gpu or cpu
        """
        self.cuda_use = cuda_use
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, \
                                batch_first=True, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.class_size)
        self.attention = Attention(hidden_size)

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,\
                function=F.log_softmax, teacher_forcing_ratio=0,\
                cuda_use=False, class_list=None, vocab_dict=None):
        """
        parameter : inputs : target sequence variable
        parameter : encoder_hidden : hidden of encoder
        parameter : encoder_outputs : outputs of encoder
        parameter : function : softmax function
        parameter : teacher_forcing_ratio : ratio of teacher_force

        """
        self.cuda_use = cuda_use
        self.class_list = class_list
        self.vocab_dict = vocab_dict

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        batch_size = encoder_outputs.size(0)

        pad_var = torch.LongTensor([self.sos_id]*batch_size) # marker
        pad_var = Variable(pad_var.view(batch_size, 1))#.cuda() # marker
        if self.cuda_use:
            pad_var = pad_var.cuda()
        decoder_init_hidden = encoder_hidden
        max_length = inputs.size(1)
        if use_teacher_forcing:
            ''' all steps together'''
            inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
            inputs = inputs[:, :-1] # batch x seq_len
            decoder_inputs = inputs 
            return self.forward_normal_teacher(decoder_inputs, decoder_init_hidden, encoder_outputs,\
                                                             function)
        else:
            decoder_input = pad_var#.unsqueeze(1) # batch x 1
            return self.forward_normal_no_teacher(decoder_input, decoder_init_hidden, encoder_outputs,\
                                                  max_length, function)
    def decode(self, step, step_output):
        '''
        step_output: batch x classes , prob_log
        symbols: batch x 1
        '''
        symbols = step_output.topk(1)[1] 
        return symbols
    def forward_step(self, input_var, hidden, encoder_outputs, function):
        '''
        normal forward, step by step or all steps together
        '''
        
        if len(input_var.size()) == 1:
            input_var = torch.unsqueeze(input_var,1)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(\
                            output.contiguous().view(-1, self.hidden_size)), dim=1)\
                            .view(batch_size, output_size, -1)
        return predicted_softmax, hidden#, #attn
    def forward_normal_teacher(self, decoder_inputs, decoder_init_hidden, encoder_outputs, function):
        decoder_outputs_list = []
        sequence_symbols_list = []
        #attn_list = []
        decoder_hidden = decoder_init_hidden
        seq_len = decoder_inputs.size(1)
        for di in range(seq_len):
            decoder_input = decoder_inputs[:, di]
            decoder_output, decoder_hidden = self.forward_step(decoder_input, 
                                                                decoder_hidden, 
                                                                encoder_outputs, 
                                                                function=function)
            step_output = decoder_output.squeeze(1)
            symbols = self.decode(di, step_output)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list#, attn_list
    def forward_normal_no_teacher(self, decoder_input, decoder_init_hidden, encoder_outputs,\
                                                 max_length,  function):
        '''
        decoder_input: batch x 1
        decoder_output: batch x 1 x classes,  probility_log
        '''
        decoder_outputs_list = []
        sequence_symbols_list = []
        #attn_list = []
        decoder_hidden = decoder_init_hidden
        for di in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, 
                                                                decoder_hidden, 
                                                                encoder_outputs, 
                                                                function=function)
            step_output = decoder_output.squeeze(1)
            symbols = self.decode(di, step_output)
            decoder_input = self.symbol_norm(symbols)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list
    def symbol_norm(self, symbols):
        symbols = symbols.view(-1).data.cpu().numpy() 
        new_symbols = []
        for idx in symbols:
            new_symbols.append(self.vocab_dict[self.class_list[idx]])
        new_symbols = Variable(torch.LongTensor(new_symbols)) 
        new_symbols = torch.unsqueeze(new_symbols, 1)
        if self.cuda_use:
            new_symbols = new_symbols.cuda()
        return new_symbols
class DNSdecoder(nn.Module):
    def __init__(self, vocab_size, class_size, emb_size, hidden_size, \
                n_layers, sos_id, eos_id, input_dropout_p, dropout_p, \
                cell_name='lstm',cuda_use=False):
        super(DNSdecoder, self).__init__()
        """
        parameter : vocab_size : size of vocabulary
        parameter : class_size : size of symbols
        parameter : emb_size   : size of embedding vector
        parameter : hidden_size: size of hidden layer
        parameter : dropout_p  : rate of rnn's dropout
        parameter : n_layers   : number of hidden layer
        parameter : input_dropout_p : rate of dropout after embedding
        parameter : sos_id     :
        parameter : eos_id     :
        parameter : cell_name  : type of rnn, expect gru or lstm
        parameter " cuda_use   : true or false to control using gpu or cpu
        """
        self.cuda_use = cuda_use
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.class_size = class_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.cell_name = cell_name
        if self.cell_name == 'lstm':
            self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, \
                                batch_first=True, dropout=dropout)
        elif self.cell_name == 'gru':
            self.rnn=nn.GRU(emb_size, hidden_size, n_layers, \
                                batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, class_size)
        #self.attention = Attention(hidden_size)
    
    def forward(self, inputs,encoder_hidden=None, encoder_outputs=None,teacher_forcing_ratio=1,use_rule=True,\
                function=F.log_softmax, cuda_use=False, class_list=None, class_dict=None, vocab_list=None, vocab_dict=None):
        """
        parameter : inputs         : target sequence variable
        parameter : encoder_hidden : hidden of encoder
        parameter : encoder_outputs : outputs of encoder
        parameter : function       : softmax function
        parameter : teacher_forcing_ratio : ratio of teacher_force
        parameter : use_rule       : true or false to control genarating based rule
        """
        self.cuda_use = cuda_use
        self.class_dict = class_dict
        self.class_list = class_list
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.use_rule = use_rule

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        batch_size = encoder_outputs.size(0)

        pad_var = torch.LongTensor([self.sos_id]*batch_size) # marker

        pad_var = Variable(pad_var.view(batch_size, 1))#.cuda() # marker
        if self.cuda_use:
            pad_var = pad_var.cuda()

        decoder_init_hidden = encoder_hidden

        max_length = inputs.size(1)

        
        if use_teacher_forcing:
            ''' all steps together'''
            inputs = torch.cat((pad_var, inputs), 1)  # careful concate  batch x (seq_len+1)
            x = inputs
            inputs = inputs[:, :-1]  # batch x seq_len
            x = inputs
            
            decoder_inputs = inputs
            return self.forward_normal_teacher(decoder_inputs, decoder_init_hidden, encoder_outputs,\
                                                             function)
        else:
            decoder_input = pad_var#.unsqueeze(1) # batch x 1
            return self.forward_normal_no_teacher(decoder_input, decoder_init_hidden, encoder_outputs,\
                                                    max_length, function)
    
    def forward_step(self, input_var, hidden, encoder_outputs, function):
        '''
        normal forward, step by step or all steps together
        '''
        
        if len(input_var.size()) == 1:
            input_var = torch.unsqueeze(input_var,1)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded,hidden)

        predicted_softmax = function(self.out(\
                            output.contiguous().view(-1, self.hidden_size)), dim=1)\
                            .view(batch_size, output_size, -1)
        return predicted_softmax, hidden#, #attn
    
    def forward_normal_teacher(self, decoder_inputs, decoder_init_hidden, encoder_outputs, function):
        decoder_outputs_list = []
        sequence_symbols_list = []
        decoder_hidden = decoder_init_hidden
        seq_len = decoder_inputs.size(1)
        for di in range(seq_len):
            decoder_input = decoder_inputs[:, di]
            decoder_output, decoder_hidden = self.forward_step(decoder_input, 
                                                                decoder_hidden, 
                                                                encoder_outputs, 
                                                                function=function)
            step_output = decoder_output.squeeze(1)
            if self.use_rule == False:
                symbols = self.decode(di, step_output)
            else:
                symbols = self.decode_rule(di, sequence_symbols_list, step_output)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list#, attn_list
    def forward_normal_no_teacher(self, decoder_input, decoder_init_hidden, encoder_outputs,\
                                                 max_length,  function):
        '''
        decoder_input: batch x 1
        decoder_output: batch x 1 x classes,  probility_log
        '''
        decoder_outputs_list = []
        sequence_symbols_list = []
        decoder_hidden = decoder_init_hidden
        for di in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, 
                                                                decoder_hidden, 
                                                                encoder_outputs, 
                                                                function=function)
            step_output = decoder_output.squeeze(1)
            if self.use_rule == False:
                symbols = self.decode(di, step_output)
            else:
                symbols = self.decode_rule(di, sequence_symbols_list, step_output) 
            
            decoder_input = self.symbol_norm(symbols)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list
    
    def decode(self, step, step_output):
        '''
        step_output: batch x classes , prob_log
        symbols: batch x 1
        '''
        symbols = step_output.topk(1)[1] 
        return symbols
    def symbol_norm(self, symbols):
        symbols = symbols.view(-1).data.cpu().numpy() 
        new_symbols = []
        for idx in symbols:
            new_symbols.append(self.vocab_dict[self.class_list[idx]])
        new_symbols = Variable(torch.LongTensor(new_symbols)) 
        new_symbols = torch.unsqueeze(new_symbols, 1)
        if self.cuda_use:
            new_symbols = new_symbols.cuda()
        return new_symbols
    
    def decode_rule(self, step, sequence_symbols_list, step_output):
        symbols = self.rule_filter(sequence_symbols_list, step_output)
        return symbols
    
    def rule1_filter(self):
        filters = []
        filters.append(self.class_dict['+'])
        filters.append(self.class_dict['-'])
        filters.append(self.class_dict['*'])
        filters.append(self.class_dict['/'])
        filters.append(self.class_dict['^'])
        filters.append(self.class_dict[')'])
        filters.append(self.class_dict['END_token'])
        return np.array(filters)
    def rule2_filter(self):
        filters = []
        for idx, symbol in enumerate(self.class_list):
            if 'temp' in symbol or symbol in ['PI', '(']:
                filters.append(self.class_dict[symbol])
            else:
                try:
                    float(symbol)
                    filters.append(self.class_dict[symbol])
                except:
                    pass
        return np.array(filters)
    def rule4_filter(self):
        filters = []
        filters.append(self.class_dict['('])
        filters.append(self.class_dict[')'])
        filters.append(self.class_dict['+'])
        filters.append(self.class_dict['-'])
        filters.append(self.class_dict['*'])
        filters.append(self.class_dict['/'])
        filters.append(self.class_dict['^'])
        filters.append(self.class_dict['END_token'])
        return np.array(filters)
    def rule5_filter(self):
        filters = []
        for idx, symbol in enumerate(self.class_list):
            if 'temp' in symbol or symbol in ['PI', '(', ')']:
                filters.append(self.class_dict[symbol])
            else:
                try:
                    float(symbol)
                    filters.append(self.class_dict[symbol])
                except:
                    pass
        return np.array(filters)
    def rule_filter(self, sequence_symbols_list, current):
        '''
        32*28
        '''
        op_list = ['+','-','*','/','^']
        cur_out = current.cpu().data.numpy()
        #print len(sequence_symbols_list)
        #pdb.set_trace()
        cur_symbols = []
        if sequence_symbols_list == []:
            #filters = self.filter_op()
            filters = np.append(self.filter_op(), self.filter_END())
            for i in range(cur_out.shape[0]):
                cur_out[i][filters] = -float('inf')
                cur_symbols.append(np.argmax(cur_out[i]))
        else:
            for i in range(sequence_symbols_list[0].size(0)):
                symbol = sequence_symbols_list[-1][i].cpu().data[0]
                if self.class_list[symbol] in ['+','-','*','/','^']:
                    filters = self.rule1_filter()
                    cur_out[i][filters] = -float('inf')
                elif 'temp' in self.class_list[symbol] or self.class_list[symbol] in ['PI']:
                    filters = self.rule2_filter()
                    cur_out[i][filters] = -float('inf')
                elif self.class_list[symbol] in ['(']:
                    filters = self.rule4_filter()
                    cur_out[i][filters] = -float('inf')
                elif self.class_list[symbol] in [')']:
                    filters = self.rule5_filter()
                    cur_out[i][filters] = -float('inf')
                else:
                    try:
                        float(self.class_dict[symbol])
                        filters = self.rule2_filter()
                        cur_out[i][filters] = -float('inf')
                    except:
                        pass
                cur_symbols.append(np.argmax(cur_out[i]))
        cur_symbols = Variable(torch.LongTensor(cur_symbols))
        cur_symbols = torch.unsqueeze(cur_symbols, 1)
        if self.cuda_use:
            cur_symbols = cur_symbols.cuda()
        return cur_symbols
    def filter_op(self):
        filters = []
        filters.append(self.class_dict['+']) 
        filters.append(self.class_dict['-']) 
        filters.append(self.class_dict['*']) 
        filters.append(self.class_dict['/']) 
        filters.append(self.class_dict['^']) 
        return np.array(filters)

    def filter_END(self):
        filters = []
        filters.append(self.class_dict['END_token']) 
        return np.array(filters)
