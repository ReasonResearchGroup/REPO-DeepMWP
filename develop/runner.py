#import random
import os
import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .data.process import math23k_data_processing, load_processed23k_data
from .dataload.DataLoad import math23kDataLoader
from .evaluate.evaluator import Evaluator
from .loss.Loss import NLLLoss
from .model.encode import *
from .model.decode import *
from .model.seq2seq import *
from .config import args
def train_addition(model, data_loader, train_data, test_data, evaluator,
                    loss, lr, batchsize, filename, cuda_use, postfix):
    epoch = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_ans_acc = 0
    print("---start train addition---")
    for epo in epoch:
        model.train()
        for step,data in enumerate(train_data):
            model.train()
            outputs_list, decoder_hidden, sequence_symbols_list = model(input_variable = data['train_var'],
                                                                    input_lengths = data['train_len'],
                                                                    target_variable = data['target_var'],
                                                                    use_cuda = cuda_use)
            target_var = data_loader._convert_f_e_2_d_sybmbol(data['target_var'])
            loss.reset()
            for i, output in enumerate(outputs_list):
            # cuda step_output  =  step_output.cuda()
                target = target_var[:, i]
                if cuda_use:
                    target = target.cuda()
                    output = output.cuda()
                loss.eval_batch(output.contiguous().view(batchsize, -1), target)
            #loss_total += loss.get_loss()
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            #print("\rstep{}:loss={}".format(step + 1, loss.get_loss()), end='')
        model.eval()
        train_temp_acc, train_ans_acc = evaluator.evaluate(model = model,
                                                            datas = train_data,
                                                            total_num=len(data_loader.train_data) + len(data_loader.valid_data),
                                                            batch_size = batchsize,
                                                            evaluate_type = 0,
                                                            mode = 0,
                                                            post_flag=postfix)
        test_temp_acc, test_ans_acc =evaluator.evaluate(model = model,
                                                        datas = test_data,
                                                        total_num = len(data_loader.test_data),
                                                        batch_size = batchsize,
                                                        evaluate_type = 0,
                                                        mode = 0,
                                                        post_flag = postfix)
        if test_ans_acc>best_ans_acc:
            best_ans_acc = test_ans_acc
            state = {"epoch":epo+1,
                    "valid_acc":"{} / {}".format(train_temp_acc,train_ans_acc),
                    "test_acc":"{} / {}".format(test_temp_acc,test_ans_acc),
                    "model_state_dict": model.state_dict()}
            path = "./experiment/train_add_best_epoch_" + filename + ".pkl"
            torch.save(state,path)
        
        print("epoch[{}]:\nvalid_tem_acc:{} / valid_ans_acc:{}\ntest_tem_acc:{} / test_ans_acc:{}".format(
                    epo+1,  train_temp_acc,   train_ans_acc,    test_temp_acc,      test_ans_acc))
    
    path = "./experiment/train_add_best_epoch_" + filename + ".pkl"
    state = torch.load(path)
    best_model = Mathen(encoder,decoder,data_loader)
    best_model.load_state_dict(state['model_state_dict'])
    if cuda_use:
        best_model.cuda
    
    '''testing'''
    print("---start testing for train addition---")
    path = "./experiment/train_add_test_result_" + filename + ".json"
    best_model.eval()
    test_temp_acc, test_ans_acc = evaluator.evaluate_with_result(model = best_model,
                                                        datas = test_data,
                                                        total_num = len(data_loader.test_data),
                                                        batch_size = batchsize,
                                                        evaluate_type = 0,
                                                        mode = 0,
                                                        filename = path,
                                                        post_flag = postfix)
    print("testing:\ntest_tem_acc:{} / test_ans_acc:{}".format(test_temp_acc, test_ans_acc))
    
def Mathen_Runner(epoch, batchsize, lr, dropout, input_dropout, cuda_use=False, rule_1=True, rule_2=True, postfix=True, filename=''):
    """
    giving the epoch, batchsize, learning rate, dropout, and input dropout to implement training process of math EN model.
    ---------------------
    optional parameter:
    cuda_use : true or false to control using gpu or cpu.
    rule_1   : true or false to control using rule 1 for equation template.
    rule_2   : true or false to control using rule 2 for equation template.
    postfix  : true or false to control using postfix for equation template.
    filename : the file to save your model.
    """
    '''data process'''
    train_data, valid_data, test_data = math23k_data_processing(rule_1, rule_2, postfix)
    '''data loader'''
    data_loader = math23kDataLoader(train_data, test_data, valid_data)
    '''encoder'''
    encoder=MathenEncoder(vocab_size=data_loader.vocab_len,
                          emb_size = 128,
                          hidden_size = 512,
                          input_dropout_p = input_dropout,
                          dropout_p = dropout,
                          n_layers = 2,
                          bidirectional = True,
                          variable_lengths = True)
    '''decoder'''
    decoder=MathenDecoder(vocab_size = data_loader.vocab_len,
                                class_size = data_loader.decode_classes_len,
                                emb_size = 128,
                                hidden_size = 1024,
                                n_layers = 2,
                                sos_id = data_loader.vocab_2_ind['END_token'],
                                eos_id = data_loader.vocab_2_ind['END_token'],
                                input_dropout_p = input_dropout,
                                dropout_p = dropout)
    '''RNN model'''
    model = Mathen(encoder, decoder, data_loader)
    
    if cuda_use:
        model.cuda()
    weight = torch.ones(data_loader.decode_classes_len)
    pad = data_loader.decode_classes_2_ind['PAD_token']
    '''loss'''
    if cuda_use:
        weight = weight.cuda()
        
    loss = NLLLoss(weight, pad)
    '''evaluator'''
    evaluator = Evaluator(vocab_dict = data_loader.vocab_2_ind,
                                   vocab_list = data_loader.vocab_list,
                                   decode_classes_dict = data_loader.decode_classes_2_ind,
                                   decode_classes_list = data_loader.decode_classes_list,
                                   cuda_use=cuda_use)
    '''optimizer'''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    '''load data'''
    train_data = data_loader.loading("train", batchsize, postfix)
    test_data = data_loader.loading("test", batchsize, postfix)
    valid_data = data_loader.loading("valid", batchsize, postfix)
    
    best_ans_acc = -1
    '''training'''
    print("---start training---")
    for epo in range(epoch):
        loss_total = 0
        model.train()
        for step,data in enumerate(train_data):
            model.train()
            outputs_list, decoder_hidden, sequence_symbols_list = model(input_variable = data['train_var'],
                                                                    input_lengths = data['train_len'],
                                                                    target_variable = data['target_var'],
                                                                    use_cuda = cuda_use)
            target_var = data_loader._convert_f_e_2_d_sybmbol(data['target_var'])
            loss.reset()
            for i, output in enumerate(outputs_list):
                target = target_var[:, i]
                if cuda_use:
                    target = target.cuda()
                    output = output.cuda()
                loss.eval_batch(output.contiguous().view(batchsize, -1), target)
            loss_total += loss.get_loss()
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            print("\rstep{}:loss={}".format(step + 1, loss.get_loss()), end='')
            
        model.eval()
        valid_temp_acc, valid_ans_acc = evaluator.evaluate(model = model,
                                                            datas = valid_data,
                                                            total_num = len(data_loader.valid_data),
                                                            batch_size = batchsize,
                                                            evaluate_type = 0,
                                                            mode = 0,
                                                            post_flag=postfix)
        test_temp_acc, test_ans_acc =evaluator.evaluate(model = model,
                                                        datas = test_data,
                                                        total_num = len(data_loader.test_data),
                                                        batch_size = batchsize,
                                                        evaluate_type = 0,
                                                        mode = 0,
                                                        post_flag = postfix)
        if valid_ans_acc>best_ans_acc:
            best_ans_acc = valid_ans_acc
            state = {"epoch":epo+1,
                    "valid_acc":"{} / {}".format(valid_temp_acc,valid_ans_acc),
                    "test_acc":"{} / {}".format(test_temp_acc,test_ans_acc),
                    "model_state_dict":model.state_dict()}
            torch.save(state, filename)
        
        print("epoch[{}]:\nvalid_tem_acc:{} / valid_ans_acc:{}\ntest_tem_acc:{} / test_ans_acc:{}".format(
                    epo+1,  valid_temp_acc,   valid_ans_acc,    test_temp_acc,      test_ans_acc))
    
    '''load best model'''
    state = torch.load(filename)
    best_model = Mathen(encoder, decoder, data_loader)
    best_model.load_state_dict(state['model_state_dict'])
    if cuda_use:
        best_model.cuda
    
    '''testing'''
    print("---start testing---")
    path = "./experiment/test_result_" + filename + ".json"
    best_model.eval()
    valid_temp_acc, valid_ans_acc = evaluator.evaluate(model = best_model,
                                                        datas = valid_data,
                                                        total_num = len(data_loader.valid_data),
                                                        batch_size = batchsize,
                                                        evaluate_type = 0,
                                                        mode = 0,
                                                        post_flag = postfix)
    test_temp_acc, test_ans_acc = evaluator.evaluate_with_result(model = best_model,
                                                        datas = test_data,
                                                        total_num = len(data_loader.test_data),
                                                        batch_size = batchsize,
                                                        evaluate_type = 0,
                                                        mode = 0,
                                                        filename = path,
                                                        post_flag = postfix)
    print("testing:\nvalid_tem_acc:{} / valid_ans_acc:{}\ntest_tem_acc:{} / test_ans_acc:{}".format(
        valid_temp_acc, valid_ans_acc, test_temp_acc, test_ans_acc))
    
    '''train addition'''
    train_addition(model=best_model,
                    data_loader=data_loader,
                    train_data=train_data + valid_data,
                    test_data=test_data,
                    evaluator=evaluator,
                    loss=NLLLoss(weight, pad),
                    lr=lr,
                    batchsize=batchsize,
                    filename=filename,
                    cuda_use=cuda_use,
                    postfix=postfix)

