#import random
import os
import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#from .baseRNN import BaseRNN
#from model.attention import Attention
from evaluate.evaluator import Evaluator
from loss.Nllloss import NLLLoss
from dataload.DataLoad import math23kDataLoader
from model.RNN import *
from data.process import math23k_data_process

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_loader = math23kDataLoader()
    encoder=MathenEncoder(vocab_size=data_loader.vocab_len,
                          emb_size = 128,
                          hidden_size = 512,
                          input_dropout_p = 0.3,
                          dropout_p = 0.4,
                          n_layers = 2,
                          bidirectional = True,
                          variable_lengths = True)
    decoder=MathenDecoder(vocab_size = data_loader.vocab_len,
                                class_size = data_loader.decode_classes_len,
                                emb_size = 128,
                                hidden_size = 1024,
                                n_layers = 2,
                                sos_id = data_loader.vocab_2_ind['END_token'],
                                eos_id = data_loader.vocab_2_ind['END_token'],
                                input_dropout_p = 0.3,
                                dropout_p = 0.4,)
    model=Mathen(encoder,decoder,data_loader)
    if device=="cuda":
        model.cuda()
        cuda_use=True
    else:
        cuda_use=False
    weight = torch.ones(data_loader.decode_classes_len)
    pad = data_loader.decode_classes_2_ind['PAD_token']
    loss = NLLLoss(weight, pad)

    evaluator = Evaluator(vocab_dict = data_loader.vocab_2_ind,
                                   vocab_list = data_loader.vocab_list,
                                   decode_classes_dict = data_loader.decode_classes_2_ind,
                                   decode_classes_list = data_loader.decode_classes_list,
                                   cuda_use=cuda_use)
    epoch=100
    batchsize=64
    lr=0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss=NLLLoss()
    #train_data,valid_data,test_data=math23k_data_process()
    data_loader.train_data,data_loader.valid_data,data_loader.test_data=math23k_data_process()
    train_data=data_loader.train2vec(batchsize)
    test_data=data_loader.test2vec(batchsize)
    # train_data={'train_var':train_data['train_var'][:2],'train_len':train_data['train_len'][:2],
    #     'target_var':train_data['target_var'][:2],'target_len':train_data['target_len'][:2],
    #     'sentence':train_data['sentence'][:2],'id':train_data['id'][:2],'num_list':train_data['num_list'][:2],
    #     'solution':train_data['solution'][:2]}
    # test_data={'train_var':test_data['train_var'][:2],'train_len':test_data['train_len'][:2],
    #     'target_var':test_data['target_var'][:2],'target_len':test_data['target_len'][:2],
    #     'sentence':test_data['sentence'][:2],'id':test_data['id'][:2],'num_list':test_data['num_list'][:2],
    #     'solution':test_data['solution'][:2]}
    for epo in range(epoch):
        loss_total=0
        model.train()
        for i,inputs in enumerate(train_data['train_var']):
            model.train()
            outputs_list, decoder_hidden, sequence_symbols_list=model(input_variable=inputs,
                                                                    input_lengths=train_data['train_len'][i],
                                                                    target_variable=train_data['target_var'][i],
                                                                    use_cuda=cuda_use)
            target_var_i=data_loader._convert_f_e_2_d_sybmbol(train_data['target_var'][i])
            loss.reset()
            for step, step_output in enumerate(outputs_list):
            # cuda step_output = step_output.cuda()
                target = target_var_i[:, step]
                loss.eval_batch(step_output.contiguous().view(batchsize, -1), target)
            loss_total+=loss.get_loss()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            print("\rstep{}:loss={}".format(i+1,loss.get_loss()),end='')
        model.eval()
        train_temp_acc, train_ans_acc =evaluator.evaluate(model = model,
                                                            data = train_data,
                                                            data_loader = data_loader,
                                                            total_num = len(data_loader.test_data),
                                                            batch_size = batchsize,
                                                            evaluate_type = 0,
                                                            mode = 0,
                                                            post_flag=False)
        test_temp_acc, test_ans_acc =evaluator.evaluate(model = model,
                                                            data = test_data,
                                                            data_loader = data_loader,
                                                            total_num = len(data_loader.test_data),
                                                            batch_size = batchsize,
                                                            evaluate_type = 0,
                                                            mode = 0,
                                                            post_flag=False)
        print("epoch[{}]\ntrain_tem_acc:{} / train_ans_acc:{}\ntest_tem_acc:{} / test_ans_acc:{}".format(
            epo,train_temp_acc, train_ans_acc,test_temp_acc,test_ans_acc))
        