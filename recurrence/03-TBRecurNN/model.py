import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
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
class MathenEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size=100, hidden_size=128, \
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, \
                 variable_lengths=True):
        super(MathenEncoder, self).__init__()
        '''
        parameter : vocab_size : size of vocabulary
        parameter : emb_size   : size of embedding vector
        parameter : hidden_size: size of hidden layer
        parameter : dropout_p  : rate of rnn's dropout
        parameter : n_layers   : number of hidden layer
        parameter : input_dropout_p : rate of dropout after embedding
        parameter : bidirectional   : true or false to control rnn's direction
        '''
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
class RecurNN(nn.Module):
    def __init__(self, vocab_size, class_size,op_size , emb_size=100, hidden_size=128, \
                n_layers=1, dropout_p=0, bidirectional=True, cuda_use=False):
        super(RecurNN, self).__init__()
        self.cuda_use = cuda_use
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, dropout=dropout_p, \
                            batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size, class_size)
        self.attention = Attention(hidden_size)
        self.recurcell = nn.Linear(hidden_size, op_size)
        
    def forward(self, input_var, postfix_equ, quantities):
        embedded = self.embedding(input_var, input_lenths, target_lenths)
        output, hidden = self.rnn(embedded)
        batch_size = input_var.size(0)
        
        location_list = self.quantities_locate(input_var, quantities)
        output_quantities=self.get_quantities_output(output,location_list)
        
        output, attn = self.attention(output_quantities, output)
        #------
        self.out(output.contiguous().view(-1, self.hidden_size))
        predicted_softmax = function(self.out(\
                            output.contiguous().view(-1, self.hidden_size)), dim=1)\
                            .view(batch_size, output_size, -1)

    def quantities_locate(self, input_var, quantities):
        location_list=[]
        for i, input_ in enumerate(input_var):
            quan_list = quantities[i]
            location=[]
            for symbol in quan_list:
                mask = input_.eq(symbol)
                location.append(input_[mask])
            if self.cuda_use:
                location = torch.tensor(location, device='cuda')
            else:
                location = torch.tensor(location)
            location_list.append(location)
        return location_list
    def get_quantities_output(self,hidden, location_list):
        hidden_quantities=[]
        for i, location in enumerate(location_list):
            hi = hidden[i].index_select(dim=1, index=location)
            hidden_quantities.append(hi)
        if self.cuda_use:
                hidden_quantities = torch.tensor(hidden_quantities, device='cuda')
            else:
                hidden_quantities = torch.tensor(hidden_quantities)
        return hidden_quantities
            

class TBRecurNN(nn.Module):
    def __init__(self, seq2seq, RecurNN, cuda_use,vocab_dict,decode_class_dict,op_dict):
        super(TBRecurNN, self).__init__()
        self.temp_pred_module = seq2seq
        self.ans_module = RecurNN
        self.cuda_use = cuda_use
        self.vocab_dict = vocab_dict
        self.decode_class_dict = decode_class_dict
        self.op_dict = op_dict
    def forword(self,input_var,input_lengh,target_var,target_lengh,function=F.softmax):
        output, hidden, symbol_list = self.temp_pred_module()
        postfix_equ = self.postfix_equ(symbol_list)
        quantities = self.get_quantities(postfix_equ)
        self.ans_module(input_var, postfix_equ, quantities)
        
        pass
    
    def postfix_equ(self, symbol_list):
        batch_size = symbol_list.size(0)
        seq_size = symbol_list.size(1)
        batch_equ=[]
        for i in range(batch_size):
            equ_list=[]
            for j in range(seq_size):
                equ_list.append(self.decode_class_dict[symbol_list[i][j]])
            batch_equ.append(self.postfix_equation(equ_list))
        return batch_equ
    def postfix_equation(equ_list):
        stack = []
        post_equ = []
        #op_list = ['+', '-', '*', '/', '^']
        #priori = {'^':3, '*':2, '/':2, '+':1, '-':1}
        for elem in equ_list:
            if elem == '(':
                stack.append('(')
            elif elem == ')':
                while 1:
                    op = stack.pop()
                    if op == '(':
                        break
                    else:
                        post_equ.append(op)
            elif elem == 'op':#[+-*/^]
                while 1:
                    if stack == []:
                        break
                    elif stack[-1] == '(':
                        break
                    else:
                        op = stack.pop()
                        post_equ.append(op)
                stack.append(elem)
            else:
                post_equ.append(elem)
        while stack != []:
            post_equ.append(stack.pop())
        return post_equ
    def get_quantities(self, postfix_equ):
        batch_size = len(postfix_equ)
        batch_quantities=[]
        for i in batch_size:
            quantities=[]
            for symbol in postfix_equ[i]:
                if symbol != 'op':
                    quantities.append(self.vocab_dict[symbol])
            quantities = torch.tensor(quantities)
            batch_quantities.append(quantities)
        return batch_quantities


if __name__ == "__main__":
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).reshape(2, 5)
    print(x.size())
    idx = [[0, 0], [1, 1], [2, 2]]
    idx = [(0, 0), (1, 1), (2, 2)]
    z = torch.tensor([1, 3, 4])
    print(x[0])
    #print(x.eq(5))
    mask = x.eq(torch.tensor(5))
    print(x[mask])
    
    #y=x[0,:].(dim=0,index=z)
    #y=x[0,:].index_select(dim=0,index=z)

    #y = x.eq(z)
    #print(y)
    