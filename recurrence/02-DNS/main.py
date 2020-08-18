import random
import os
#import sys
import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from process import *
from dataload import math23kDataLoader
from evaluator import Evaluator
from loss import NLLLoss
from retrieval import Retrieval
from hybrid import Hybrid
from RNN import *
from config import get_arg
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def DNS_runner(epoch, batchsize, lr, dropout, input_dropout, encodecell, decodecell, \
                cuda_use=False, rule_1=False, rule_2=False, postfix=False, filename=''):
    '''data process'''
    #train_data, valid_data, test_data = math23k_data_processing(rule_1, rule_2, postfix)
    train_data, valid_data, test_data = load_processed23k_data()
    '''data loader'''
    data_loader = math23kDataLoader(train_data,test_data)
    '''encoder'''
    encoder = DNSencoder(vocab_size=data_loader.vocab_len,
                        emb_size=128,
                        hidden_size=512,
                        input_dropout=input_dropout,
                        dropout=dropout,
                        n_layer=2,
                        cell_name=ecell)
    '''decoder'''
    decoder = DNSdecoder(vocab_size=data_loader.vocab_len,
                        class_size=data_loader.decode_classes_len,
                        emb_size=128,
                        hidden_size=512,
                        n_layers=2,
                        sos_id = data_loader.vocab_2_ind['END_token'],
                        eos_id = data_loader.vocab_2_ind['END_token'],
                        input_dropout = input_dropout,
                        dropout=dropout,
                        cell_name=dcell)
    
    '''RNN model'''
    seq2seq=DNS(encoder,decoder,data_loader,cuda_use)
    if cuda_use:
        seq2seq.cuda()
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
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=lr)
    #loss=NLLLoss()
    '''load data'''
    train_data = data_loader.loading("train", batchsize, postfix)
    test_data = data_loader.loading("test", batchsize, postfix)
    valid_data = data_loader.loading("valid", batchsize, postfix)
    #train_data = train_data + valid_data
    
    #best_v_temp_acc=-1
    best_ans_acc = -1
    '''training'''
    print("---start training---")
    for epo in range(epoch):
        loss_total = 0
        seq2seq.train()
        for step,data in enumerate(train_data):
            seq2seq.train()
            outputs_list, decoder_hidden, sequence_symbols_list = seq2seq(input_variable = data['train_var'],
                                                                    input_lengths = data['train_len'],
                                                                    target_variable=data['target_var'],
                                                                    teacher_forcing_ratio=1)
            target_var = data_loader._convert_f_e_2_d_sybmbol(data['target_var'])
            loss.reset()
            batchsize_ = len(data['id'])
            for i, output in enumerate(outputs_list):
            # cuda step_output  =  step_output.cuda()
                target = target_var[:, i]
                if cuda_use:
                    target = target.cuda()
                    output = output.cuda()
                loss.eval_batch(output.contiguous().view(batchsize_, -1), target)
            loss_total += loss.get_loss()
            
            seq2seq.zero_grad()
            loss.backward()
            optimizer.step()
            print("\rstep{}:loss={}".format(step + 1, loss.get_loss()), end='')
        seq2seq.eval()
        valid_temp_acc, valid_ans_acc = evaluator.evaluate(model = seq2seq,
                                                            datas = valid_data,
                                                            total_num = len(data_loader.valid_data),
                                                            batch_size = batchsize,
                                                            evaluate_type = 0,
                                                            mode = 0,
                                                            post_flag=postfix)
        train_temp_acc, train_ans_acc =evaluator.evaluate(model = seq2seq,
                                                        datas = train_data,
                                                        total_num = len(data_loader.train_data),
                                                        batch_size = batchsize,
                                                        evaluate_type = 0,
                                                        mode = 0,
                                                        post_flag=postfix)
        test_temp_acc, test_ans_acc,result = evaluator.evaluate_with_result(model = seq2seq,
                                                            datas = test_data,
                                                            total_num = len(data_loader.test_data),
                                                            batch_size = batchsize,
                                                            evaluate_type = 0,
                                                            mode = 0,
                                                            post_flag=postfix)
        if valid_ans_acc>best_ans_acc:
            best_ans_acc = valid_ans_acc
            state = {"epoch":epo+1,
                    "valid_acc":"{} / {}".format(valid_temp_acc,valid_ans_acc),
                    "test_acc":"{} / {}".format(test_temp_acc,test_ans_acc),
                    "model_state_dict":seq2seq.state_dict()}
            path = "./experiment/best_epoch_"+filename+".pkl"
            torch.save(state,path)
        
        print("epoch[{}]:\nvalid_tem_acc:{} / valid_ans_acc:{}\ntest_tem_acc:{} / test_ans_acc:{}".format(
                    epo + 1, valid_temp_acc, valid_ans_acc, test_temp_acc, test_ans_acc))
        if epo % 10 == 0:
                path_r='./experiment/result_'+str(epo+1)+'.json'
                write_json_data(result,path_r)
    
    # '''load best model'''
    # path = "./experiment/best_epoch_"+filename+".pkl"
    # state = torch.load(path)
    # best_model = DNS(encoder,decoder,data_loader)
    # best_model.load_state_dict(state['model_state_dict'])
    # '''retrieval model'''
    # retrieval_model = Retrieval(train_data, data_loader.vocab_list)
    # '''hybrid model'''
    # hybrid_model = Hybrid(best_model, retrieval_model, cuda_use)
    # if cuda_use:
    #     hybrid_model.cuda()
    # '''testing'''
    # test_data = data_loader.loading("test", 1, postfix)
    # valid_data = data_loader.loading("valid", 1, postfix)
    # print("---start testing---")
    # path = "./experiment/test_result_" + filename + ".json"
    # hybrid_model.eval()
    # valid_temp_acc, valid_ans_acc = evaluator.evaluate(model = hybrid_model,
    #                                                     datas = valid_data,
    #                                                     total_num = len(data_loader.valid_data),
    #                                                     batch_size = batchsize,
    #                                                     evaluate_type = 0,
    #                                                     mode = 0,
    #                                                     post_flag = postfix)
    # test_temp_acc, test_ans_acc = evaluator.evaluate_with_result(model = hybrid_model,
    #                                                     datas = test_data,
    #                                                     total_num = len(data_loader.test_data),
    #                                                     batch_size = batchsize,
    #                                                     evaluate_type = 0,
    #                                                     mode = 0,
    #                                                     filename = path,
    #                                                     post_flag=postfix)
    
def hybrid_runner(theta,epoch, batchsize, lr, dropout, input_dropout, encodecell, decodecell, \
                cuda_use=False, rule_1=False, rule_2=False, postfix=False, filename=''):
    train_data, valid_data, test_data = load_processed23k_data()
    #train_data = train_data + valid_data
    data_loader = math23kDataLoader(train_data, test_data)
    '''encoder'''
    encoder = DNSencoder(vocab_size=data_loader.vocab_len,
                        emb_size=128,
                        hidden_size=512,
                        input_dropout=input_dropout,
                        dropout=dropout,
                        n_layer=2,
                        cell_name=ecell)
    '''decoder'''
    decoder = DNSdecoder(vocab_size=data_loader.vocab_len,
                        class_size=data_loader.decode_classes_len,
                        emb_size=128,
                        hidden_size=512,
                        n_layers=2,
                        sos_id = data_loader.vocab_2_ind['END_token'],
                        eos_id = data_loader.vocab_2_ind['END_token'],
                        input_dropout = input_dropout,
                        dropout=dropout,
                        cell_name=dcell)
    '''evaluator'''
    evaluator = Evaluator(vocab_dict = data_loader.vocab_2_ind,
                                   vocab_list = data_loader.vocab_list,
                                   decode_classes_dict = data_loader.decode_classes_2_ind,
                                   decode_classes_list = data_loader.decode_classes_list,
                                   cuda_use=cuda_use)
    '''load data'''
    train_data = data_loader.loading("train", 1, postfix)
    test_data = data_loader.loading("test", 1, postfix)
    '''load best model'''
    path = "./experiment/best_epoch_"+filename+".pkl"
    state = torch.load(path)
    best_model = DNS(encoder,decoder,data_loader,cuda_use)
    best_model.load_state_dict(state['model_state_dict'])
    '''retrieval model'''
    #retrieval_model = Retrieval(train_data, data_loader.vocab_list)
    retrieval_model = Retrieval(train_data, data_loader.vocab_list, True, cuda_use, theta)
    
    '''hybrid model'''
    hybrid_model = Hybrid(best_model, retrieval_model, cuda_use)
    if cuda_use:
        hybrid_model.cuda()
    '''testing'''
    print("---start testing---")
    path = "./experiment/test_result_" + filename + ".json"
    hybrid_model.eval()
    test_temp_acc, test_ans_acc,_ = evaluator.hybrid_evaluate(model = hybrid_model,
                                                        test_data=test_data,
                                                        
                                                        #total_num = len(data_loader.test_data),
                                                        #batch_size = batchsize,
                                                        #evaluate_type = 0,
                                                        #mode = 0,
                                                        postfix = postfix)
    # test_temp_acc, test_ans_acc = evaluator.evaluate_with_result(model = hybrid_model,
    #                                                     datas = test_data,
    #                                                     total_num = len(data_loader.test_data),
    #                                                     batch_size = batchsize,
    #                                                     evaluate_type = 0,
    #                                                     mode = 0,
    #                                                     filename = path,
    #                                                     post_flag=postfix)
def retrieval_runner():
    batchsize = 256
    postfix = False
    rule_1 = False
    rule_2 = False
    cuda_use=True if torch.cuda.is_available() else False
    '''data process'''
    #train_data, valid_data, test_data = math23k_data_processing(rule_1, rule_2, postfix)
    train_data, valid_data, test_data = load_processed23k_data()
    
    datas = valid_data + train_data
    data_num = len(datas)
    test_times = int(data_num / 1000) + 1
    new_data = []
    temp_acc_total = 0
    ans_acc_total = 0
    temp_count = 0
    ans_count = 0
    
    # result_list=[]
    # for i in range(test_times):
    #     if data_num >= (i + 1) * 1000:
    #         data = datas[i * 1000 : (i + 1) * 1000]
    #     else:
    #         data = datas[i * 1000 :data_num]
    #     new_data.append(data)
    
    # for i in range(test_times):
    #     #print('\r第{}次：'.format(1 + i))
    #     k_train_data = []
    #     for j,data in enumerate(new_data):
    #         if i != j:
    #             k_train_data = k_train_data + data
    #         else:
    #             k_test_data = data
    #     '''data loader'''
    #     data_loader = math23kDataLoader(k_train_data, k_test_data)
    #     k_train_data = data_loader.loading("train", 1, postfix)
    #     k_test_data = data_loader.loading("test", 1, postfix)
    #     #valid_data = data_loader.loading("valid", 1, postfix)
    #     '''evaluator'''
    #     evaluator = Evaluator(vocab_dict = data_loader.vocab_2_ind,
    #                             vocab_list = data_loader.vocab_list,
    #                             decode_classes_dict = data_loader.decode_classes_2_ind,
    #                             decode_classes_list = data_loader.decode_classes_list,
    #                             cuda_use=False)
    #     total_len = len(k_test_data)
    #     model = Retrieval(k_train_data, data_loader.vocab_list, True)
    #     count, acc_right, id_right_and_error, result = evaluator.retrieval_evaluate(model, k_test_data, postfix)
    #     temp_acc_total = temp_acc_total + count / total_len
    #     ans_acc_total = ans_acc_total + acc_right / total_len
    #     temp_count = temp_count + count
    #     ans_count = ans_count + acc_right
    #     result_list = result_list + result
    #     print('第{}次：{} / {}\n{} / {}'.format(1 + i, count, acc_right, count / total_len, acc_right / total_len))
    # path = './experiment/retrieval_result.json'
    # with open(path, 'w', encoding='utf-8') as f:
    #     json.dump(result_list, f, indent=4)
    # print('k cross valid:{} / {} {}/\ntemp_acc:{}\nans_acc:{}'.format(temp_count, ans_count, data_num, \
    #     temp_acc_total / test_times, ans_acc_total / test_times))
    
    '''data loader'''
    data_loader = math23kDataLoader(datas, test_data)
    train_data = data_loader.loading("train", 1, postfix)
    test_data = data_loader.loading("test", 1, postfix)
    #valid_data = data_loader.loading("valid", 1, postfix)
    total_len = len(test_data)
    
    '''evaluator'''
    evaluator = Evaluator(vocab_dict = data_loader.vocab_2_ind,
                            vocab_list = data_loader.vocab_list,
                            decode_classes_dict = data_loader.decode_classes_2_ind,
                            decode_classes_list = data_loader.decode_classes_list,
                            cuda_use=False)
    #test_data = random.sample(train_data, 1000)
    '''retrieval model'''
    model = Retrieval(train_data, data_loader.vocab_list, True, cuda_use)

    '''test'''
    #count, acc_right, id_right_and_error = evaluator.retrieval_evaluate(model, test_data, postfix)
    test_data = random.sample(test_data, k=10)
    #count, acc_right, id_right_and_error, result = evaluator.retrieval_evaluate(model, test_data, postfix)
    result=evaluator.retrieval_evaluate_1(model, test_data, postfix)
    path = './experiment/retrieval_result_sample10.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    # print('\n')
    # print('total len:', total_len)
    # print('temp:{} / ans:{}'.format(count,acc_right))
    # print(count / total_len)
    # print(acc_right / total_len)
    # print(len(id_right_and_error[-1]))
    # print(len(id_right_and_error[0]))
    # print(len(id_right_and_error[1]))

def retrieval_runner2():
    batchsize = 256
    postfix = False
    rule_1 = False
    rule_2 = False
    
    '''data process'''
    train_data, valid_data, test_data = math23k_data_processing(rule_1, rule_2, postfix)
    
    datas = valid_data + train_data

    '''data loader'''
    data_loader = math23kDataLoader(datas, test_data)
    train_data = data_loader.loading("train", 1, postfix)
    test_data = data_loader.loading("test", 1, postfix)
    #valid_data = data_loader.loading("valid", 1, postfix)
    total_len = len(test_data)
    
    #test_data = random.sample(train_data, 1000)
    '''retrieval model'''
    model_r = Retrieval(train_data, data_loader.vocab_list, True)
    model_f = Retrieval(train_data, data_loader.vocab_list, False)
    path_r='./experiment/retrieval_result_r.json'
    path_f = './experiment/retrieval_result_f.json'
    f = open(path_r, 'r', encoding='utf-8')
    data_r = json.load(f)
    for data in data_r:
        text = data['tar_sentence']
        text = [word for word in text.split(' ')]
        bow_r = model_r.dictionary.doc2bow(text)
        bow_f = model_f.dictionary.doc2bow(text)
        vec_r = model_r.tfidf_model[bow_r]
        vec_f = model_f.tfidf_model[bow_f]
        vec_r = [i[1] for i in vec_r]
        vec_f = [i[1] for i in vec_f]
        print(text)
        print(vec_f)
        new_p_test = []
        for word in text:
            if 'temp' in word:
                new_p_test.append(word)
            elif 'PI' in word:
                new_p_test.append(word)
            else:
                for char in word:
                    new_p_test.append(char)
        text = new_p_test
        print(text)
        print(vec_r)
        print('\n')


def makefilename(ecell, dcell):
    filename = ecell[0] + dcell[0]
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
    ecell = args.encodecell
    dcell=args.decodecell
    filename = makefilename(ecell, dcell)
    theta=args.theta
    print("{}--{}".format(filename[0],filename[1]))
    #DNS_runner(epoch,batchsize,lr,dropout,input_dropout,ecell,dcell,cuda_use=cuda_use,filename=filename)
    hybrid_runner(epoch,batchsize,lr,dropout,input_dropout,ecell,dcell,cuda_use=cuda_use,filename=filename)
    #retrieval_runner()
    #retrieval_runner2()
    # path = './experiment/retrieval_result.json'
    # path2 = './experiment/retrieval_result1.json'
    # with open(path, 'r',encoding='utf8') as f:
    #     json_data = json.load(f)
    #     f.close()
    # with open(path2, 'w',encoding='utf8') as f:
    #     json_str = json.dumps(json_data, indent=4)
    #     f.write(json_str)
    #     f.close()
    
