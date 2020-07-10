import random

import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#from baseRNN import BaseRNN
from .attention import Attention
#from evaluator import Evaluator,NLLLoss
#from DataLoad import math23kDataLoader
class Mathen(nn.Module):
    def __init__(self, encoder, decoder,data_loader=None,device=None,decoder_function=F.log_softmax):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader=data_loader
        self.device = device
        self.decoder_function = decoder_function
        
            
    def forward(self, input_variable, input_lengths=None, target_variable=None,\
                teacher_forcing_ratio=1, mode=0, use_cuda=False):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        encoder_hidden = encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])

        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decoder_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              use_cuda = use_cuda,
                              class_list=self.dataloader.decode_classes_list,
                              vocab_dict=self.dataloader.vocab_2_ind)
        return result
    def _cat_directions(self, h):
        if self.encoder.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def inference(self, input, target):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1  
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]        # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds
class MathenEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size=100, hidden_size=128, \
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, \
                 variable_lengths=True):
        super(MathenEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        
        #self.rnn_cell = nn.GRU
        #self.embedding = embed_model
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers,
                          batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        #pdb.set_trace()
        
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True,enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

class MathenDecoder(nn.Module):
    def __init__(self, vocab_size, class_size, emb_size=100, hidden_size=128, \
                n_layers=1, sos_id=1, eos_id=0, input_dropout_p=0, dropout_p=0):
        super(MathenDecoder,self).__init__()
        self.vocab_size = vocab_size
        #self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        #self.rnn_cell = nn.LSTM
        #self.embedding = embed_model
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, \
                                 batch_first=True, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.class_size)
        self.attention = Attention(hidden_size)

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,\
                function=F.log_softmax, teacher_forcing_ratio=0,\
                use_cuda=False, class_list=None,vocab_dict=None):
        '''
        使用rule的时候，teacher_forcing_rattio = 0
        '''
        #self.use_rule = use_rule
        self.use_cuda = use_cuda
        #self.data_loader = data_loader
        self.class_list = class_list
        self.vocab_dict = vocab_dict
        ''' self.class_dict = class_dict
        self.class_list = class_list
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list '''

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        #pdb.set_trace()
        batch_size = encoder_outputs.size(0)
        #batch_size = inputs.size(0)

        pad_var = torch.LongTensor([self.sos_id]*batch_size) # marker

        pad_var = Variable(pad_var.view(batch_size, 1))#.cuda() # marker
        if self.use_cuda:
            pad_var = pad_var.cuda()

        decoder_init_hidden = encoder_hidden

        # if template_flag == False:
        #     max_length = 40
        # else:
        max_length = inputs.size(1)

        #inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
        #inputs = inputs[:, :-1] # batch x seq_len

        if use_teacher_forcing:
            ''' all steps together'''
            inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
            inputs = inputs[:, :-1] # batch x seq_len
            decoder_inputs = inputs 
            return self.forward_normal_teacher(decoder_inputs, decoder_init_hidden, encoder_outputs,\
                                                             function)
        else:
            #decoder_input = inputs[:,0].unsqueeze(1) # batch x 1
            decoder_input = pad_var#.unsqueeze(1) # batch x 1
            #pdb.set_trace()
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
            #deocder_input = torch.unsqueeze(decoder_input, 1)
            #print '1', deocder_input.size()
            decoder_output, decoder_hidden = self.forward_step(decoder_input, 
                                                                decoder_hidden, 
                                                                encoder_outputs, 
                                                                function=function)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            #if self.use_rule == False:
            symbols = self.decode(di, step_output)
            # else:
            #     symbols = self.decode_rule(di, sequence_symbols_list, step_output)
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
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            #print step_output.size()
            #if self.use_rule == False:
            symbols = self.decode(di, step_output)
            # else:
            #     symbols = self.decode_rule(di, sequence_symbols_list, step_output) 
            decoder_input = self.symbol_norm(symbols)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        #print sequence_symbols_list
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list
    def symbol_norm(self, symbols):
        symbols = symbols.view(-1).data.cpu().numpy() 
        new_symbols = []
        for idx in symbols:
            #print idx, 
            #print self.class_list[idx],
            #pdb.set_trace()
            #print self.vocab_dict[self.class_list[idx]]
            new_symbols.append(self.vocab_dict[self.class_list[idx]])
        new_symbols = Variable(torch.LongTensor(new_symbols)) 
        #print new_symbols
        new_symbols = torch.unsqueeze(new_symbols, 1)
        if self.use_cuda:
            new_symbols = new_symbols.cuda()
        return new_symbols
