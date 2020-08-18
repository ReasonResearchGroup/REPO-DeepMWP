import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import json
import pickle
import os
import sys
import random
from ..data.process_tools import postfix_equation, post_solver, solve_equation,inverse_temp_to_num
class Evaluator(object):
    def __init__(self, vocab_dict, vocab_list, decode_classes_dict,\
                    decode_classes_list, cuda_use=False):
        #self.loss = loss
        self.cuda_use = cuda_use
        # if self.cuda_use:
        #     self.loss.cuda()

        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.decode_classes_dict = decode_classes_dict
        self.decode_classes_list = decode_classes_list

        self.pad_in_classes_idx = self.decode_classes_dict['PAD_token']
        self.end_in_classes_idx = self.decode_classes_dict['END_token']

    def _convert_f_e_2_d_sybmbol(self, target_variable):
        new_variable = []
        batch,colums = target_variable.size()
        for i in range(batch):
            tmp = [0]*colums
            for j in range(colums):
                #print target_variable[i][j].data[0]
                try:
                  idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].item()]]
                  #tmp.append(idx)
                  tmp[j] = idx
                except:
                  pass
                  #print self.vocab_list[target_variable[i][j].data[0]].encode('utf-8')
                #tmp.append(idx)
            new_variable.append(tmp)
        try:
            return Variable(torch.LongTensor(np.array(new_variable)))
            #print ('-------',new_variable)
        except:
            print ('-+++++++',new_variable)

    def get_new_tempalte(self, seq_var, num_list):
        equ_list = []
        for idx in seq_var.data.cpu().numpy():
            if idx == self.pad_in_classes_idx:
                break
            equ_list.append(self.decode_classes_list[idx])
        equ_list = inverse_temp_to_num(equ_list, num_list)
        try:
            equ_list = equ_list[:equ_list.index('END_token')]
        except:
            pass
        return equ_list

    def compute_gen_ans(self, seq_var, num_list, post_flag):
        equ_list = []
        for idx in seq_var.data.cpu().numpy():
            if idx == self.pad_in_classes_idx:
                break
            equ_list.append(self.decode_classes_list[idx])
        try:
            equ_list = inverse_temp_to_num(equ_list, num_list)
        except:
            return 'inverse error'
        try:
            equ_list = equ_list[:equ_list.index('END_token')]
        except:
            pass
        try:
            if post_flag == False:
                ans = solve_equation(equ_list)
            else:
                ans = post_solver(equ_list)
            float(ans)
            return ans
        except:
            return ("compute error")

    def evaluate(self, model, datas, total_num, batch_size, evaluate_type,
                    mode, post_flag=False, name_save='train',data_loader=None):
        if evaluate_type == 0:
            teacher_forcing_ratio = 0.0
        else:
            teacher_forcing_ratio = 1.0

        count = 0
        acc_right = 0

        id_right_and_error = {1:[], 0:[], -1:[]} 
        id_template = {}
        pg_total_list = []
        xxx = 0

        for i,data in enumerate(datas):
            input_lengths = data['train_len']
            input_variables = Variable(torch.LongTensor(data['train_var']))

            if self.cuda_use:
                input_variables = input_variables.cuda()
            
            batch_index = data['id']
            batch_text = data['sentence']
            batch_num_list = data['num_list']
            batch_solution = data['solution']
            batch_size = len(batch_index)

            #if template_flag == True:
            target_variables = data['target_var']
            target_lengths = data['target_len']

            target_variables = Variable(torch.LongTensor(target_variables))
            if self.cuda_use:
                target_variables = target_variables.cuda()
            


            decoder_outputs, decoder_hidden, symbols_list = model(
                                                input_variable = input_variables,
                                                input_lengths = input_lengths,
                                                target_variable = target_variables,
                                                teacher_forcing_ratio = teacher_forcing_ratio)


            seqlist = symbols_list
            seq_var = torch.cat(seqlist, 1)
            batch_pg = []
            
            for i in range(batch_size):
                wp_index = batch_index[i]
                p_list = []
                for j in range(len(decoder_outputs)): 
                    mm_elem_idx = seq_var[i][j].cpu().data.numpy().tolist()
                    if mm_elem_idx == self.end_in_classes_idx:
                        break
                    num_p = decoder_outputs[j][i].topk(1)[0].cpu().data.numpy()[0]
                    p_list.append(str(num_p))
                    
                batch_pg.append(p_list)
                
            for i in range(batch_size):
                target_equ = target_variables[i].cpu().data.numpy()
                tmp_equ = []
                if print_flag_:
                    print (target_equ)
                for target_idx in target_equ:
                    elem = self.vocab_list[target_idx]
                    if elem =='END_token':
                        break
                    tmp_equ.append(elem)
                
                gen_ans = self.compute_gen_ans(seq_var[i], batch_num_list[i], post_flag)
                gen_equ = self.get_new_tempalte(seq_var[i], batch_num_list[i])
                target_ans = batch_solution[i]
                
                if 'error' in gen_ans:
                    id_right_and_error[-1].append(batch_index[i])
                    continue
                else:
                    if abs(float(gen_ans) - float(target_ans)) <1e-5:
                        acc_right += 1
                        id_right_and_error[1].append(batch_index[i])
                    else:
                        id_right_and_error[0].append(batch_index[i])
                        continue 

            target_variables = self._convert_f_e_2_d_sybmbol(target_variables)
            if self.cuda_use:
                target_variables = target_variables.cuda()
            for i in range(batch_size):
                right_flag = 0
                for j in range(target_variables.size(1)):
                    if seq_var[i][j].item() == self.end_in_classes_idx and \
                            target_variables[i][j].item() == self.end_in_classes_idx:
                        right_flag = 1
                        break
                    if target_variables[i][j].item() != seq_var[i][j].item():
                        break
                count += right_flag
        
        print  ('\n--------',acc_right, total_num)
        return count*1.0/total_num, acc_right*1.0/total_num
                
    def evaluate_with_result(self, model,datas,total_num, batch_size, \
                      evaluate_type, mode,filename='', post_flag=False,data_loader=None ):
        if evaluate_type == 0:
            teacher_forcing_ratio = 0.0
        else:
            teacher_forcing_ratio = 1.0

        count = 0
        acc_right = 0

        id_right_and_error = {1:[], 0:[], -1:[]} 
        id_template = {}
        pg_total_list = []
        xxx = 0
        for step,data in enumerate(datas):
            input_lengths = data['train_len']
            input_variables = Variable(torch.LongTensor(data['train_var']))

            if self.cuda_use:
                input_variables = input_variables.cuda()
            
            batch_index = data['id']
            batch_text = data['sentence']
            batch_num_list = data['num_list']
            batch_solution = data['solution']
            target_variables = data['target_var']
            target_lengths = data['target_len']
            batch_size = len(batch_index)
            
            if self.cuda_use:
                target_variables = target_variables.cuda()
            
            decoder_outputs, decoder_hidden, symbols_list = model(
                                        input_variable = input_variables,
                                        input_lengths = input_lengths,
                                        target_variable = target_variables,
                                        teacher_forcing_ratio = teacher_forcing_ratio)
            seqlist = symbols_list
            seq_var = torch.cat(seqlist, 1)
            batch_pg = []
            for i in range(batch_size):
                wp_index = batch_index[i]
                p_list = []
                for j in range(len(decoder_outputs)): 
                    
                    mm_elem_idx = seq_var[i][j].cpu().data.numpy().tolist()
                    
                    if mm_elem_idx == self.end_in_classes_idx:
                        break
                    num_p = decoder_outputs[j][i].topk(1)[0].cpu().data.numpy()[0]
                    p_list.append(str(num_p))
                batch_pg.append(p_list)
            
            target_equ_right_flag=[]
            t_variables = self._convert_f_e_2_d_sybmbol(target_variables)
            if self.cuda_use:
                t_variables = t_variables.cuda()
            for i in range(batch_size):
                right_flag = 0
                for j in range(t_variables.size(1)):
                    if seq_var[i][j].item() == self.end_in_classes_idx and \
                            t_variables[i][j].item() == self.end_in_classes_idx:
                        right_flag = 1
                        break
                    if t_variables[i][j].item() != seq_var[i][j].item():
                        break
                target_equ_right_flag.append(right_flag)
                count += right_flag

            for i in range(batch_size):
                target_equ = target_variables[i].cpu().data.numpy()
                tmp_equ = []
                if print_flag_:
                    print (target_equ)
                for target_idx in target_equ:
                    elem = self.vocab_list[target_idx]
                    if elem =='END_token':
                        break
                    tmp_equ.append(elem)

                gen_ans = self.compute_gen_ans(seq_var[i], batch_num_list[i], post_flag)
                gen_equ = self.get_new_tempalte(seq_var[i], batch_num_list[i])
                target_ans = batch_solution[i]
                tmp_equ = inverse_temp_to_num(tmp_equ, batch_num_list[i])
                if 'error' in gen_ans:
                    ans_right_flag=-1
                else:
                    if abs(float(gen_ans) - float(target_ans)) <1e-5:
                        acc_right += 1
                        ans_right_flag = 1
                    else:
                        ans_right_flag = 0
                pg_total_list.append(dict({'index': batch_index[i], 'gen_equ': gen_equ, \
                        'equ':tmp_equ,'gen_ans': gen_ans, 'ans': str(target_ans),
                        'temp_right_flag':target_equ_right_flag[i],'ans_right_flag':ans_right_flag}))
        if filename != '':
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(pg_total_list, f,indent=4)
        print  ('\n--------',acc_right, total_num)
        return count*1.0/total_num, acc_right*1.0/total_num

    def retrieval_evaluate(self, model, test_data,postfix):
        id_right_and_error = {1: [], 0: [], -1: []}
        acc_right = 0
        count = 0
        #print('\n')
        random_result=[]
        for step, data in enumerate(test_data):
            print('\r{}'.format(step),end='')
            similarity, gen_temp, sentence = model.js(data['sentence'][0])
            tar_sentence = data['sentence']
            
            target_variables = data['target_var']
            batch_num_list = data['num_list']
            batch_solution = data['solution']
            batch_index = data['id']
            seqlist = gen_temp
            seq_var = seqlist
            
            target_equ_right_flag = []
            seq_var=self._convert_f_e_2_d_sybmbol(seq_var)
            t_variables = self._convert_f_e_2_d_sybmbol(target_variables)
            if self.cuda_use:
                t_variables = t_variables.cuda()
            for i in range(1):
                right_flag = 0
                for j in range(t_variables.size(1)):
                    if seq_var[i][j].item() == self.end_in_classes_idx and \
                            t_variables[i][j].item() == self.end_in_classes_idx:
                        right_flag = 1
                        break
                    if t_variables[i][j].item() != seq_var[i][j].item():
                        break
                target_equ_right_flag.append(right_flag)
                count += right_flag
            for i in range(1):
                target_equ = target_variables[i].cpu().data.numpy()
                tmp_equ = []
                for target_idx in target_equ:
                    elem = self.vocab_list[target_idx]
                    if elem =='END_token':
                        break
                    tmp_equ.append(elem)
                
                gen_ans = self.compute_gen_ans(seq_var[i], batch_num_list[i], postfix)
                gen_equ = self.get_new_tempalte(seq_var[i], batch_num_list[i])
                target_ans = batch_solution[i]
                gen_equ = []
                gen_temp = gen_temp[i].cpu().data.numpy()
                for target_idx in gen_temp:
                    elem = self.vocab_list[target_idx]
                    if elem =='END_token':
                        break
                    gen_equ.append(elem)
                if 'error' in gen_ans:
                    id_right_and_error[-1].append(batch_index[i])
                    continue
                else:
                    if abs(float(gen_ans) - float(target_ans)) <1e-5:
                        acc_right += 1
                        id_right_and_error[1].append(batch_index[i])
                    else:
                        id_right_and_error[0].append(batch_index[i])
                        continue
            if random.random() > 0.85:
                result = {'choosed_sentence': sentence, 'tar_sentence': tar_sentence[0], 'choosed_equ': gen_equ, 'tar_equ': tmp_equ}
                random_result.append(result)
        return count,acc_right,id_right_and_error,random_result
    def retrieval_evaluate_1(self, model, test_data, postfix):
        result=[]
        for step, data in enumerate(test_data):
            print('\r{}'.format(step),end='')
            tar_sentence = data['sentence']
            
            target_variables = data['target_var']
            batch_num_list = data['num_list']
            batch_solution = data['solution']
            batch_index = data['id']
            batch_equ=data['equ'][0]

            top5data = model.js(data['sentence'][0])
        
            r = {}
            r['choosed'] = top5data
            r['target'] = {'text':tar_sentence[0],'equ':batch_equ}
            result.append(r)
        return result
    def hybrid_evaluate(self, model, test_data, postfix):
        id_right_and_error = {1: [], 0: [], -1: []}
        acc_right = 0
        count = 0
        #print('\n')
        random_result=[]
        for step, data in enumerate(test_data):
            print('\r{}'.format(step),end='')
            tar_sentence = data['sentence']

            input_lengths = data['train_len']
            input_variables = Variable(torch.LongTensor(data['train_var']))
            
            target_variables = data['target_var']
            target_variables=torch.LongTensor(target_variables)
            batch_num_list = data['num_list']
            batch_solution = data['solution']
            batch_index = data['id']
            
            solver_type,gen_temp = model(data['sentence'][0],input_variables,input_lengths,target_variables)
            seqlist = gen_temp
            seq_var = seqlist
            target_equ_right_flag = []
            seq_var=self._convert_f_e_2_d_sybmbol(seq_var)
            t_variables = self._convert_f_e_2_d_sybmbol(target_variables)
            if self.cuda_use:
                t_variables = t_variables.cuda()
            for i in range(1):
                right_flag = 0
                for j in range(t_variables.size(1)):
                    if seq_var[i][j].item() == self.end_in_classes_idx and \
                            t_variables[i][j].item() == self.end_in_classes_idx:
                        right_flag = 1
                        break
                    if t_variables[i][j].item() != seq_var[i][j].item():
                        break
                target_equ_right_flag.append(right_flag)
                count += right_flag
            for i in range(1):
                target_equ = target_variables[i].cpu().data.numpy()
                tmp_equ = []
                for target_idx in target_equ:
                    elem = self.vocab_list[target_idx]
                    if elem =='END_token':
                        break
                    tmp_equ.append(elem)
                
                gen_ans = self.compute_gen_ans(seq_var[i], batch_num_list[i], postfix)
                gen_equ = self.get_new_tempalte(seq_var[i], batch_num_list[i])
                target_ans = batch_solution[i]
                gen_equ = []
                gen_temp = gen_temp[i].cpu().data.numpy()
                for target_idx in gen_temp:
                    elem = self.vocab_list[target_idx]
                    if elem =='END_token':
                        break
                    gen_equ.append(elem)
                if 'error' in gen_ans:
                    id_right_and_error[-1].append(batch_index[i])
                    continue
                else:
                    if abs(float(gen_ans) - float(target_ans)) <1e-5:
                        acc_right += 1
                        id_right_and_error[1].append(batch_index[i])
                    else:
                        id_right_and_error[0].append(batch_index[i])
                        continue
            if random.random() > 0.85:
                result = {'choosed_sentence': sentence, 'tar_sentence': tar_sentence[0], 'choosed_equ': gen_equ, 'tar_equ': tmp_equ}
                random_result.append(result)
                #print(tar_sentence[0],tmp_equ)
                #print(sentence,gen_equ)
                # print('\n')
        return count,acc_right,id_right_and_error,random_result