import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import json
import pickle# as pickle
import os
import sys
def postfix_equation(equ_list):
    stack = []
    post_equ = []
    op_list = ['+', '-', '*', '/', '^']
    priori = {'^':3, '*':2, '/':2, '+':1, '-':1}
    for elem in equ_list:
        elem = str(elem)
        if elem == '(':
            stack.append('(')
        elif elem == ')':
            while 1:
                op = stack.pop()
                if op == '(':
                    break
                else:
                    post_equ.append(op)
        elif elem in op_list:
            while 1:
                if stack == []:
                    break
                elif stack[-1] == '(':
                    break
                elif priori[elem] > priori[stack[-1]]:
                    break
                else:
                    op = stack.pop()
                    post_equ.append(op)
            stack.append(elem)
        else:
            #if elem == 'PI':
            #    post_equ.append('3.14')
            #else:
            #    post_equ.append(elem)
            post_equ.append(elem)
    while stack != []:
        post_equ.append(stack.pop())
    return post_equ

def post_solver(post_equ):
    stack = [] 
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        elem = str(elem)
        if elem not in op_list:
            op_v = elem
            if '%' in op_v:
                op_v = float(op_v[:-1])/100.0
            stack.append(str(op_v))
        elif elem in op_list:
            op_v_1 = stack.pop()
            op_v_1 = float(op_v_1)
            op_v_2 = stack.pop()
            op_v_2 = float(op_v_2)
            if elem == '+':
                stack.append(str(op_v_2+op_v_1))
            elif elem == '-':
                stack.append(str(op_v_2-op_v_1))
            elif elem == '*':
                stack.append(str(op_v_2*op_v_1))
            elif elem == '/':
                stack.append(str(op_v_2/op_v_1))
            else:
                stack.append(str(op_v_2**op_v_1))
    return stack.pop()

def solve_equation(equ_list):
    post_equ = postfix_equation(equ_list)
    ans = post_solver(post_equ)
    return ans

print_flag_ = 0

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


    def inverse_temp_to_num(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem:
                index = alphabet.index(elem[-1])
                new_equ_list.append(num_list[index])
            elif 'PI' == elem:
                new_equ_list.append('3.14')
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def inverse_temp_to_num_(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem:
                index = alphabet.index(elem[-1])
                try:
                    new_equ_list.append(str(num_list[index]))
                except:
                    return []
            elif 'PI' == elem:
                new_equ_list.append('3.14')
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def get_new_tempalte(self, seq_var, num_list):
        equ_list = []
        for idx in seq_var.data.cpu().numpy():
            if idx == self.pad_in_classes_idx:
                break
            equ_list.append(self.decode_classes_list[idx])
        equ_list = self.inverse_temp_to_num_(equ_list, num_list)
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
        #equ_list = equ_list[2:]
        try:
            #print equ_list[:equ_list.index('END_token')]
            #print equ_list, num_list,
            equ_list = self.inverse_temp_to_num(equ_list, num_list)
        except:
            #print "inverse error",
            return 'inverse error'
        try:
            equ_list = equ_list[:equ_list.index('END_token')]
            #print num_list
            #print ' '.join(equ_list),
            #if '1' in equ_list:
            #    print 'g1'
            #else:
            #    print 
        except:
            #print equ_list
            pass
        #equ_string = ''.join(equ_list)
        #equ_list = split_equation(equ_string)
        if 0:
            print (num_list)
            print (equ_list)
        try:
            #ans = solve_equation(equ_list)
            if print_flag_:
                print ('---debb-',num_list)
                print ('---debb-',equ_list)
            if post_flag == False:
                ans = solve_equation(equ_list)
            else:
                ans = post_solver(equ_list)
            if 0:
                print (ans)
                print () 
            float(ans)
            return ans
        except:
            if 0:
                print ("compute error", post_flag)
                print 
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
            #batch_size = len(batch_index)

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
                                                teacher_forcing_ratio = teacher_forcing_ratio,
                                                mode = mode,
                                                use_cuda = self.cuda_use)


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
                
                pg_total_list.append(dict({'index': batch_index[i], 'gen_equ': gen_equ, \
                        'pg':batch_pg[i], 'gen_ans': gen_ans, 'ans': target_ans}))
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
        with open("./data/pg_seq_norm_"+str(post_flag)+"_"+name_save+".json", 'w') as f:
            json.dump(pg_total_list, f)
        print  ('\n--------',acc_right, total_num)
        return count*1.0/total_num, acc_right*1.0/total_num
                
    def evaluate_with_result(self, model,datas,total_num, batch_size, \
                      evaluate_type, mode,filename, post_flag=False,data_loader=None ):
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
            #batch_truth = data['truth']
            target_variables = data['target_var']
            target_lengths = data['target_len']
            #target_variables = Variable(torch.LongTensor(target_variables))
            if self.cuda_use:
                target_variables = target_variables.cuda()
            
            decoder_outputs, decoder_hidden, symbols_list = model(
                                        input_variable = input_variables,
                                        input_lengths = input_lengths,
                                        target_variable = target_variables,
                                        teacher_forcing_ratio = teacher_forcing_ratio,
                                        mode = mode,
                                        use_cuda = self.cuda_use)
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
                tmp_equ=self.inverse_temp_to_num_(tmp_equ,batch_num_list[i])
                if 'error' in gen_ans:
                    ans_right_flag=-1
                    #id_right_and_error[-1].append(batch_index[i])
                else:
                    if abs(float(gen_ans) - float(target_ans)) <1e-5:
                        acc_right += 1
                        #id_right_and_error[1].append(batch_index[i])
                        ans_right_flag = 1
                    else:
                        #id_right_and_error[0].append(batch_index[i])
                        ans_right_flag = 0
                pg_total_list.append(dict({'index': batch_index[i], 'gen_equ': gen_equ, \
                        'equ':tmp_equ,'gen_ans': gen_ans, 'ans': str(target_ans),
                        'temp_right_flag':target_equ_right_flag[i],'ans_right_flag':ans_right_flag}))
                
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(pg_total_list, f,indent=4)
        print  ('\n--------',acc_right, total_num)
        return count*1.0/total_num, acc_right*1.0/total_num
    def _init_rl_state(self, encoder_hidden, bi_flag):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h, bi_flag) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden, bi_flag)
        return encoder_hidden

    def _cat_directions(self, h, bi_flag):
        if bi_flag:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        h = torch.cat(h[0:h.size(0)], 1)
        return h

    def _write_data_json(self, data, filename):
        with open(filename, 'wb') as f:
            json.dump(data, f)

    def _write_data_pickle(self, data, filename):
        with open(filename, 'w') as f:
            pickle.dump(data, f)
