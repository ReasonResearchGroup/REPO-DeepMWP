#import random
import os
#import sys
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
from data.process import math23k_data_processing
from config import get_arg
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def Mathen_Runner(epoch, batchsize, lr, dropout, input_dropout, cuda_use=False, rule_1=True, rule_2=True, postfix=True, filename=''):
    '''data process'''
    train_data, valid_data, test_data = math23k_data_processing(rule_1, rule_2, postfix, filename)
    '''data loader'''
    data_loader = math23kDataLoader(train_data,valid_data,test_data)
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
    #loss=NLLLoss()
    '''load data'''
    train_data = data_loader.loading("train", batchsize, postfix)
    test_data = data_loader.loading("test", batchsize, postfix)
    valid_data = data_loader.loading("valid", batchsize, postfix)
    train_data = train_data + valid_data
    
    #best_v_temp_acc=-1
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
            # cuda step_output  =  step_output.cuda()
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
                                                            data_loader = data_loader,
                                                            total_num = len(valid_data)*batchsize,
                                                            batch_size = batchsize,
                                                            evaluate_type = 0,
                                                            mode = 0,
                                                            post_flag=postfix)
        test_temp_acc, test_ans_acc =evaluator.evaluate(model = model,
                                                        datas = test_data,
                                                        data_loader = data_loader,
                                                        total_num = len(test_data)*batchsize,
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
            save_model(state,filename)
        
        print("epoch[{}]:\nvalid_tem_acc:{} / valid_ans_acc:{}\ntest_tem_acc:{} / test_ans_acc:{}".format(
                    epo+1,  valid_temp_acc,   valid_ans_acc,    test_temp_acc,      test_ans_acc))
    '''testing'''
    print("---start testing---")
    state = load_model(filename)
    best_model = Mathen(encoder,decoder,data_loader)
    best_model.load_state_dict(state['model_state_dict'])
    if cuda_use:
        best_model.cuda
    best_model.eval()
    valid_temp_acc, valid_ans_acc = evaluator.evaluate(model = best_model,
                                                        datas = valid_data,
                                                        data_loader = data_loader,
                                                        total_num = len(valid_data)*batchsize,
                                                        batch_size = batchsize,
                                                        evaluate_type = 0,
                                                        mode = 0,
                                                        post_flag = postfix)
    test_temp_acc, test_ans_acc = evaluator.evaluate_with_result(model = best_model,
                                                        datas = test_data,
                                                        data_loader = data_loader,
                                                        total_num = len(test_data)*batchsize,
                                                        batch_size = batchsize,
                                                        evaluate_type = 0,
                                                        mode = 0,
                                                        filename = filename,
                                                        post_flag = postfix)
    print("testing:\nvalid_tem_acc:{} / valid_ans_acc:{}\ntest_tem_acc:{} / test_ans_acc:{}".format(
        valid_temp_acc,valid_ans_acc,test_temp_acc,test_ans_acc))
def save_model(state,filename):
    path = "./experiment/best_epoch_"+filename+".pkl"
    torch.save(state,path)
def load_model(filename):
    path = "./experiment/best_epoch_"+filename+".pkl"
    state = torch.load(path)
    return state
def make_filename(rule1,rule2,postfix):
    if rule1:
        a = "1"
    else:
        a = "0"
    if rule2:
        b = "1"
    else:
        b = "0"
    if postfix:
        c = "1"
    else:
        c = "0"
    filename = a+b+c
    return filename
if __name__ == "__main__":
    cuda_use = True  if torch.cuda.is_available() else False
    print("cuda_use:{}".format(cuda_use))
    
    args = get_arg()
    
    batchsize = args.batchsize
    dropout = args.dropout
    input_dropout = args.dropout
    epoch = args.epoch
    lr = args.lr
    postfix = args.postfix == "True"
    rule_1 = args.rule_1 == "True"
    rule_2 = args.rule_2 == "True"
    filename = make_filename(rule_1, rule_2, postfix)
    print(rule_1, rule_2, postfix)
    
    Mathen_Runner(epoch, batchsize, lr, dropout, input_dropout, cuda_use, rule_1, rule_2, postfix, filename)
    
    
    # train_data={'train_var':train_data['train_var'][:2],'train_len':train_data['train_len'][:2],
    #     'target_var':train_data['target_var'][:2],'target_len':train_data['target_len'][:2],
    #     'sentence':train_data['sentence'][:2],'id':train_data['id'][:2],'num_list':train_data['num_list'][:2],
    #     'solution':train_data['solution'][:2]}
    # test_data={'train_var':test_data['train_var'][:2],'train_len':test_data['train_len'][:2],
    #     'target_var':test_data['target_var'][:2],'target_len':test_data['target_len'][:2],
    #     'sentence':test_data['sentence'][:2],'id':test_data['id'][:2],'num_list':test_data['num_list'][:2],
    #     'solution':test_data['solution'][:2]}
    
        