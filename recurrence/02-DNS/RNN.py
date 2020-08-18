import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

class DNSencoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, input_dropout, dropout, n_layer, cell_name='gru'):
        super(DNSencoder,self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size,)
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.cell_name = cell_name
        if self.cell_name == 'gru':
            self.rnn = nn.GRU(emb_size, hidden_size, n_layer, batch_first=True)
        elif self.cell_name == 'lstm':
            self.rnn=nn.LSTM(emb_size, hidden_size, n_layer, batch_first=True)
    
    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        #print(embedded.size())
        embedded = self.input_dropout(embedded)
        #print(embedded.size())
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
class DNSdecoder(nn.Module):
    def __init__(self, vocab_size, class_size, emb_size, hidden_size, \
                n_layers, sos_id, eos_id, input_dropout, dropout, cell_name='lstm'):
        super(DNSdecoder, self).__init__()
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
                function=F.log_softmax, use_cuda=False, class_list=None, class_dict=None,vocab_list=None,vocab_dict=None):
        self.use_cuda = use_cuda
        self.class_dict = class_dict
        self.class_list = class_list
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.use_rule = use_rule

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        batch_size = encoder_outputs.size(0)

        pad_var = torch.LongTensor([self.sos_id]*batch_size) # marker

        pad_var = Variable(pad_var.view(batch_size, 1))#.cuda() # marker
        if self.use_cuda:
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
            #print(decoder_inputs.device)
            return self.forward_normal_teacher(decoder_inputs, decoder_init_hidden, encoder_outputs,\
                                                             function)
        else:
            #decoder_input = inputs[:,0].unsqueeze(1) # batch x 1
            decoder_input = pad_var#.unsqueeze(1) # batch x 1
            #pdb.set_trace()
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

        #output, attn = self.attention(output, encoder_outputs)
        #y=output.contiguous().view(-1, self.hidden_size)
        #x=self.out(output.contiguous().view(-1, self.hidden_size))
        #predicted_softmax = function(x, dim=1).view(batch_size, output_size, -1)
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
            #symbols = self.decode(di, step_output)
            # else:
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
            # if self.use_rule == False:
            #     symbols = self.decode(di, step_output)
            # else:
            symbols = self.decode_rule(di, sequence_symbols_list, step_output) 
            
            decoder_input = self.symbol_norm(symbols)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        #print sequence_symbols_list
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
        if self.use_cuda:
            new_symbols = new_symbols.cuda()
        return new_symbols
    
    def decode_rule(self, step, sequence_symbols_list, step_output):
        symbols = self.rule_filter(sequence_symbols_list, step_output)
        return symbols
    
    def rule_filter_(self, sequence_symbols_list, current):
        '''
        32*28
        '''
        op_list = ['+','-','*','/','^']
        cur_out = current.cpu().data.numpy()
        #print len(sequence_symbols_list)
        #pdb.set_trace()
        cur_symbols = []
        if sequence_symbols_list == [] or len(sequence_symbols_list) <= 1:
            #filters = self.filter_op()
            filters = np.append(self.filter_op(), self.filter_END())
            for i in range(cur_out.shape[0]):
                cur_out[i][filters] = -float('inf')
                cur_symbols.append(np.argmax(cur_out[i]))
        else:
            for i in range(sequence_symbols_list[0].size(0)):
                num_var = 0
                num_op = 0
                for j in range(len(sequence_symbols_list)):
                    symbol = sequence_symbols_list[j][i].cpu().data[0]
                    if self.class_list[symbol] in op_list:
                        num_op += 1
                    elif 'temp' in self.class_list[symbol] or self.class_list[symbol] in ['1', 'PI']:
                        num_var += 1
                if num_var >= num_op + 2:
                    filters = self.filter_END()
                    cur_out[i][filters] = -float('inf')
                elif num_var == num_op + 1:
                    filters = self.filter_op() 
                    cur_out[i][filters] = -float('inf')
                cur_symbols.append(np.argmax(cur_out[i]))
        cur_symbols = Variable(torch.LongTensor(cur_symbols))
        cur_symbols = torch.unsqueeze(cur_symbols, 1)
        if self.use_cuda:
            cur_symbols = cur_symbols.cuda()
        return cur_symbols
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
        if self.use_cuda:
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
class DNS(nn.Module):
    def __init__(self, encoder, decoder, data_loader,cuda_use, decoder_function=F.log_softmax):
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
        #print(encoder_hidden.size())
        #print(encoder_outputs.device)
        if self.encoder.cell_name == self.decoder.cell_name:
            pass
        elif self.encoder.cell_name=='gru' and self.decoder.cell_name=='lstm':
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif self.encoder.cell_name == 'lstm' and self.decoder.cell_name == 'gru':
            encoder_hidden = encoder_hidden[0]
        #print(encoder_outputs.device)
        result = self.decoder(inputs=target_variable,
                            encoder_hidden=encoder_hidden,
                            encoder_outputs=encoder_outputs,
                            function=self.decoder_function,
                            use_cuda = self.cuda_use,
                            class_list=self.dataloader.decode_classes_list,
                            vocab_dict=self.dataloader.vocab_2_ind,
                            class_dict=self.dataloader.decode_classes_2_ind,
                            vocab_list=self.dataloader.vocab_list,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            use_rule=use_rule)
        return result
    
class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        '''
        output: decoder,  (batch, 1, hiddem_dim2)
        context: from encoder, (batch, n, hidden_dim1)
        actually, dim2 == dim1, otherwise cannot do matrix multiplication 
        '''
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (b, o, dim) * (b, dim, i) -> (b, o, i)
        attn = torch.bmm(output, context.transpose(1,2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (b, o, i) * (b, i, dim) -> (b, o, dim)
        mix = torch.bmm(attn, context)

        combined = torch.cat((mix, output), dim=2)

        output = F.tanh(self.linear_out(combined.view(-1, 2*hidden_size)))\
                            .view(batch_size, -1, hidden_size)

        # output: (b, o, dim)
        # attn  : (b, o, i)
        return output, attn

if __name__ == "__main__":
    x = 'tt'
    try:
        float(x)
    except:
        print(1)