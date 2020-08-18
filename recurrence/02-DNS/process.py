import jieba
import numpy as np
import json
import random
import copy
def read_data_json(filename):
    #with open(filename, 'r') as f:
    f = open(filename, 'r')
    return json.load(f)
def read_math23k_json(filename):
    data_list = []
    #with open(filename, 'r',encoding="utf-8") as f:
    f = open(filename, 'r', encoding="utf-8")
    count = 0
    string = ''
    for line in f:
        count += 1
        string += line
        if count % 7 == 0:
            data_list.append(json.loads(string))
            string = ''
    return data_list

def joint_number(text_list):
    new_list = []
    i = 0
    while i<len(text_list):
        if text_list[i] == '(' and i+4<len(text_list) and text_list[i+4] == ')':
            sub = ''.join(text_list[i:i+5])
            new_list.append(sub)
            i = i+5
        else:
            new_list.append(text_list[i])
            i += 1
    return new_list

def is_number(word):
    if word[0] == '(' and word[-1] == ')':
        for elem_char in word:
            if (elem_char.isdigit()):
                return True
        return False
    if '(' in word and ')' in word and '/' in word and not word[-1].isdigit():
        for elem_char in word:
            if (elem_char.isdigit()):
                return True
        return False
        #return True
    if word[-1] == '%' and len(word)>1:
        return True
    if word[0].isdigit():
        return True
    if word[-1].isdigit():
        return True
    try:
        float(word)
        return True
    except:
        return False
    
def split_num_and_unit(word):
    num = ''
    unit = ''
    #print(">>>>>>>>>>>>>>",i)
    for idx in range(len(word)):                                                                                                                                                                                                                                              
        char = word[idx]
        if char.isdigit() or char in ['.', '/', '(', ')']:
            num += char
        else:
            unit += char
    return num, unit#.encode('utf-8')

def mask_text(seg_text_list):
    alpha = "abcdefghijklmnopqrstuvwxyz"
    count = 0
    num_dict = {}
    new_text = []
    for word in seg_text_list:
        unit = ''
        if is_number(word):
            if len(set(alpha)&set(word.lower()))>0:
                num, unit = split_num_and_unit(word)
                word = num
            try:
                num_dict[word]
            except: 
                num_dict[word] = "temp_"+alpha[count]
                count += 1
            
            #count+=1
            new_text.append(num_dict[word])
            
            #if "%" in word:
            #   new_text.append("%")
            if unit != '':
                new_text.append(unit)
        else:
            new_text.append(word)
    return num_dict,new_text
def mask_equ(equ_str,num_dict):
    if '3.14%' not in equ_str and '3.1416' not in equ_str:
        equ_str = equ_str.replace('3.14', '&PI&', 15)
    num_dict = sorted(num_dict.items(), key=lambda x: len(x[0]), reverse=True)
    for k,v in num_dict:
        equ_str = equ_str.replace(k, "&{}&".format(v), 15)
    equ_list = []
    num_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%', '.']
    for elem in equ_str.split("&"):
        if "temp_" in elem or "PI" in elem:
            equ_list.append(elem)
        else:
            start = ''
            for char in elem:
                if char not in num_set:
                    if start != '':
                        equ_list.append(start)
                    equ_list.append(char)
                    start = ''
                else:
                    start += char
            if start != '':
                equ_list.append(start)
    new_equ_list = []
    for elem_equ in equ_list:
        if elem_equ == '[':
            elem_equ = '('
        if elem_equ == ']':
            elem_equ = ')'
        new_equ_list.append(elem_equ)
    equ_list = new_equ_list[:]

    return equ_list

def num_list_processed(num_list):
    new_num_list = []
    for num in num_list:
        if '%' in num:
            new_num_list.append(float(num[:-1])*1.0/100)
        elif '(' in num:
            new_num_list.append(eval(num))
        
        else:
            num_, _ = split_num_and_unit(num)
            new_num_list.append(float(num_))
    return new_num_list

def norm_equation(equ_list):
    new_list=[]
    i=0
    while i<len(equ_list):
        if (i + 4) < len(equ_list) and 'temp' in equ_list[i] and '+' in equ_list[i + 1] and 'temp' in equ_list[i + 2] and '+' in equ_list[i + 3] and 'temp' in equ_list[i + 4]:
            if i - 1 >= 0 and equ_list[i - 1] in ['/', '-', '*']:
                new_list.append(equ_list[i])
                i += 1
                continue
            if i + 5 < len(equ_list) and equ_list[i + 5] in ['/', '-', '*']:
                new_list.append(equ_list[i])
                i += 1
                continue  
            temp = [equ_list[i], equ_list[i + 2], equ_list[i + 4]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1] + ['+'] + sort_temp[1:2] + ['+'] + sort_temp[2:3]
            new_list += new_temp
            i += 5
        elif (i+4)<len(equ_list) and 'temp' in equ_list[i] and '*' in equ_list[i+1] and 'temp' in equ_list[i+2] and '*' in equ_list[i+3] and 'temp' in equ_list[i+4]:
            if i - 1 >= 0 and equ_list[i - 1] in ['/', '-']:
                new_list.append(equ_list[i])
                i += 1
                continue
            if i + 5 < len(equ_list) and equ_list[i + 5] in ['/', '-']:
                new_list.append(equ_list[i])
                i += 1
                continue  
            temp = [equ_list[i], equ_list[i+2], equ_list[i+4]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]+['*']+sort_temp[2:3]
            new_list += new_temp
            i += 5
        elif (i + 2) < len(equ_list) and 'temp' in equ_list[i] and '+' in equ_list[i + 1] and 'temp' in equ_list[i + 2]:
            if i - 1 >= 0 and equ_list[i - 1] in ['/', '-', '*']:
                new_list.append(equ_list[i])
                i += 1
                continue
            if i + 3 < len(equ_list) and equ_list[i + 3] in ['/', '-', '*']:
                new_list.append(equ_list[i])
                i += 1
                continue  
            temp = [equ_list[i], equ_list[i+2]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['+']+sort_temp[1:2]
            new_list += new_temp
            i += 3
        elif (i + 2) < len(equ_list) and 'temp' in equ_list[i] and '*' in equ_list[i + 1] and 'temp' in equ_list[i + 2]:
            if i - 1 >= 0 and equ_list[i - 1] in ['/', '-']:
                new_list.append(equ_list[i])
                i += 1
                continue
            if i + 3 < len(equ_list) and equ_list[i + 3] in ['/', '-']:
                new_list.append(equ_list[i])
                i += 1
                continue  
            temp = [equ_list[i], equ_list[i+2]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]
            new_list += new_temp
            i += 3
        else:
            new_list.append(equ_list[i])
            i += 1
    return new_list

def postfix_equation(equ_list):
    stack = []
    post_equ = []
    op_list = ['+', '-', '*', '/', '^']
    priori = {'^':3, '*':2, '/':2, '+':1, '-':1}
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
        elif elem in op_list:#[+-*/^]
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
            post_equ.append(elem)
    while stack != []:
        post_equ.append(stack.pop())
    return post_equ

def inverse_temp_to_num(equ_list, num_list):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new_equ_list = []
    for elem in equ_list:
        if 'temp' in elem:
            index = alphabet.index(elem[-1])
            new_equ_list.append(str(num_list[index]))
        elif 'PI' == elem:
            new_equ_list.append('3.14')
        else:
            new_equ_list.append(elem)
    return new_equ_list

def list2str(equ_list):
    equ_str = ''
    for elem in equ_list:
        """ if 'temp' in elem or 'PI' in elem:
            equ_str+="&{}&".format(elem)
        else: """
        equ_str += elem
    return equ_str

def post_solver(post_equ):
    stack = []
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        if elem not in op_list:
            op_v = elem
            
            stack.append(str(op_v))
        elif elem in op_list:
            op_v_1 = stack.pop()
            op_v_1 = float(op_v_1)
            if stack == []:
                if elem == '-':
                    stack.append(str(0-op_v_1))
                break
            else:
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

def ans_num_joint(word):
    i = 0
    new = []
    str_ = ''
    while i<len(word):
        if word[i].isdigit() or word[i] in ['.','-']:
            str_ += word[i]
        else:
            if str_ != '':
                new.append(str_)
                str_ = ''
            new.append(word[i])
        i += 1
    return solve_equation(new)

def solve_equation(equ_list):
    if '=' in equ_list:
        equ_list = equ_list[2:]
    
    post_equ = postfix_equation(equ_list)
    ans = post_solver(post_equ)
    return ans

def ans_decimal_exception(word):
    word = str(word)
    ind = word.find('(')
    word = word[:ind]+'+'+word[ind:]
    return ans_num_joint(word)

def ans_process(word):
    try:
        float(word)
        return float(word)
    except:
        if '%' in str(word):
            return float(word[:-1])/100
        if str(word)[0] == '(' and str(word)[-1] == ')':
            return ans_num_joint(word)
        if str(word)[0] != '(' and str(word)[-1] == ')':
            return ans_decimal_exception(word)
    return -float('inf')

def rule1_stats(datas):
    rule_1 = []
    for data in datas:
        temp_data = data
        text = data['original_text']
        equ_str = data["equation"]
        '''word cut'''
        text = jieba.cut(text, cut_all=False)
        origin_text = ' '.join(text)
        
        word_list = origin_text.split(' ')
        word_list = joint_number(word_list)
        
        '''mask'''
        num_dict,new_text = mask_text(word_list)
        equ_list = mask_equ(equ_str, num_dict)
        if '千' in equ_list:
            equ_list = equ_list[:equ_list.index('千')]
        rule_1.append(equ_list)
    samples = random.sample(range(10, 100), k=16)
    random.shuffle(samples)
    ans_dict = {}
    for equ_list in rule_1:
        new_equ = inverse_temp_to_num(equ_list, samples)
        new_equ = list2str(new_equ)
        new_equ = new_equ.replace("^","**",10)
        try:
            ans = eval(new_equ[2:])
        except:
            ans = float("inf")
        try:
            ans_dict[ans].append(equ_list)
        except:
            ans_dict[ans] = []
            ans_dict[ans].append(equ_list)
    class_list = []
    for k,v in ans_dict.items():
        class_list.append(v)
    for i in range(50):
        samples = random.sample(range(10, 100), k=16)
        random.shuffle(samples)
        class_copy = copy.deepcopy(class_list)
        class_list = []
        for equ_lists in class_copy:
            ans_dict = {}
            for equ_list in equ_lists:
                new_equ = inverse_temp_to_num(equ_list, samples)
                new_equ = list2str(new_equ)
                new_equ = new_equ.replace("^","**",10)
                try:
                    ans = eval(new_equ[2:])
                except:
                    ans = float("inf")
                try:
                    ans_dict[ans].append(equ_list)
                except:
                    ans_dict[ans] = []
                    ans_dict[ans].append(equ_list)
            for k,v in ans_dict.items():
                class_list.append(v)
    class_copy = copy.deepcopy(class_list)
    class_list = []
    for equ_lists in class_copy:
        class_list_temp = []
        for equ_list in equ_lists:
            if equ_list not in class_list_temp:
                class_list_temp.append(equ_list)
        class_list.append(class_list_temp)
    return class_list

def min_list_len(equ_lists):
    equ_lists = sorted(equ_lists, key=lambda x: len(x), reverse=False)
    return equ_lists[0]

def math23k_data_process():
    train_data = read_math23k_json('./data/math23k/math23k_train.json')
    test_data = read_math23k_json('./data/math23k/math23k_test.json')
    #data_sni = read_data_json(r'data\math23k\sni_DNS.json')
    
    new_train_data = []
    new_test_data = []
    new_valid_data = []
    rule1_list = rule1_stats(train_data)
    for d in train_data:
        new_data, _, _, _ = data_process(d, rule1_list)
        if new_data:
            new_train_data.append(new_data)
    rule1_list=rule1_stats(test_data)
    for d in test_data:
        new_data,_,_,_=data_process(d,rule1_list)
        if new_data:
            new_test_data.append(new_data)
    
    new_valid_data=new_train_data[:1000]
    new_train_data=new_train_data[1000:]
    return new_train_data,new_valid_data,new_test_data

def post_data_process(data, rule1_list=None):
    temp_data = data
    temp_data = copy.deepcopy(data)
    text = data['original_text']
    equ_str = data["equation"]
    '''word cut'''
    text = jieba.cut(text, cut_all=False)
    origin_text = ' '.join(text)
    
    word_list = origin_text.split(' ')
    word_list = joint_number(word_list)
    
    '''mask'''
    num_dict,new_text = mask_text(word_list)
    equ_list = mask_equ(equ_str, num_dict)
    
    '''num_list'''
    num_list = list(num_dict.keys())
    num_list = num_list_processed(num_list)
    
    if '千' in equ_list:
        equ_list = equ_list[:equ_list.index('千')]
    
    '''postfix'''
    post = postfix_equation(equ_list)
    
    '''answer'''
    post_equ_list = inverse_temp_to_num(post, num_list)
    ans = post_solver(post_equ_list[2:])
    
    if abs(float(ans) - float(ans_process(data['ans']))) < 1e-4:
        temp_data['new_split'] = origin_text
        temp_data['text'] = ' '.join(new_text)
        temp_data["target_norm_post_template"] = post
        temp_data["target_template"] = equ_list
        temp_data["num_list"] = num_list
        temp_data["answer"] = float(ans_process(data['ans']))
        return temp_data,None
    else:
        return None,None
def rule_1_post_data_process(data,rule1_list):
    temp_data = data
    temp_data = copy.deepcopy(data)
    text = data['original_text']
    equ_str = data["equation"]
    '''word cut'''
    text = jieba.cut(text, cut_all=False)
    origin_text = ' '.join(text)
    
    word_list = origin_text.split(' ')
    word_list = joint_number(word_list)
    
    '''mask'''
    num_dict,new_text = mask_text(word_list)
    equ_list = mask_equ(equ_str,num_dict)
    
    '''num_list'''
    num_list = list(num_dict.keys())
    num_list = num_list_processed(num_list)
    
    if '千' in equ_list:
        equ_list = equ_list[:equ_list.index('千')]
    
    '''rule 1'''
    for equ_lists in rule1_list:
        if equ_list in equ_lists:
            rule_1 = min_list_len(equ_lists)
    
    '''postfix'''
    post = postfix_equation(rule_1)
    
    '''answer'''
    post_equ_list = inverse_temp_to_num(post,num_list)
    ans = post_solver(post_equ_list[2:])
    
    if abs(float(ans) - float(ans_process(data['ans']))) < 1e-4:
        temp_data['new_split'] = origin_text
        temp_data['text'] = ' '.join(new_text)
        temp_data["target_norm_post_template"] = post
        temp_data["target_template"] = rule_1
        temp_data["num_list"] = num_list
        temp_data["answer"] = float(ans_process(data['ans']))
        if rule_1 != equ_list:
            equ_str = list2str(equ_list)
            rule_1_str = list2str(rule_1)
            change_data = {'id': data['id'], 'before_norm': equ_str, 'rule_1': rule_1_str}
            return temp_data, change_data
        else:
            return temp_data, None
    else:
        return None, None
def rule_2_post_data_process(data, rule1_list=None):
    temp_data = data
    temp_data = copy.deepcopy(data)
    text = data['original_text']
    equ_str = data["equation"]
    '''word cut'''
    text = jieba.cut(text, cut_all=False)
    origin_text = ' '.join(text)
    
    word_list = origin_text.split(' ')
    word_list = joint_number(word_list)
    
    '''mask'''
    num_dict,new_text = mask_text(word_list)
    equ_list = mask_equ(equ_str,num_dict)
    
    '''num_list'''
    num_list = list(num_dict.keys())
    num_list = num_list_processed(num_list)
    
    if '千' in equ_list:
        equ_list = equ_list[:equ_list.index('千')]
    
    '''rule 2'''
    rule_2 = norm_equation(equ_list)

    '''postfix'''
    post = postfix_equation(rule_2)
    
    '''answer'''
    post_equ_list  =  inverse_temp_to_num(post,num_list)
    ans = post_solver(post_equ_list[2:])
    
    if abs(float(ans) - float(ans_process(data['ans']))) < 1e-4:
        temp_data['new_split'] = origin_text
        temp_data['text'] = ' '.join(new_text)
        temp_data["target_norm_post_template"] = post
        temp_data["target_template"] = rule_2
        temp_data["num_list"] = num_list
        temp_data["answer"] = float(ans_process(data['ans']))
        if rule_2 != equ_list:
            equ_str = list2str(equ_list)
            rule_2_str = list2str(rule_2)
            change_data = {'id': data['id'], 'before_norm': equ_str, 'rule_2': rule_2_str}
            return temp_data, change_data
        else:
            return temp_data, None
    else:
        return None, None
def data_process(data,rule1_list):
    
    temp_data = data
    temp_data = copy.deepcopy(data)
    text = data['original_text']
    equ_str = data["equation"]
    '''word cut'''
    text = jieba.cut(text, cut_all=False)
    origin_text = ' '.join(text)
    
    word_list = origin_text.split(' ')
    word_list = joint_number(word_list)
    
    '''mask'''
    num_dict,new_text = mask_text(word_list)
    equ_list = mask_equ(equ_str,num_dict)
    
    '''num_list'''
    num_list = list(num_dict.keys())
    num_list = num_list_processed(num_list)
    
    if '千' in equ_list:
        equ_list = equ_list[:equ_list.index('千')]
    
    '''rule 1'''
    for equ_lists in rule1_list:
        if equ_list in equ_lists:
            rule_1 = min_list_len(equ_lists)
    
    '''rule 2'''
    rule_2 = norm_equation(rule_1)
    
    '''rule 2 + postfix'''
    rule_2_post = postfix_equation(rule_2)
    
    '''answer'''
    post_equ_list = inverse_temp_to_num(rule_2_post,num_list)
    ans = post_solver(post_equ_list[2:])
    
    if abs(float(ans) - float(ans_process(data['ans']))) < 1e-4:
        temp_data['new_split'] = origin_text
        temp_data['text'] = ' '.join(new_text)
        temp_data["target_norm_post_template"] = rule_2_post
        temp_data["target_template"] = rule_2
        temp_data["num_list"] = num_list
        temp_data["answer"] = float(ans_process(data['ans']))
        if equ_list != rule_1 or rule_1 != rule_2:
            equ_str = list2str(equ_list)
            rule_1_str = list2str(rule_1)
            rule_2_str = list2str(rule_2)
            change_data = {'id': data['id'], 'before_norm': equ_str, 'rule_1': rule_1_str, 'rule_2': rule_2_str}
            return temp_data, change_data
        else:
            return temp_data, None
    else:
        return None, None
def math23k_data_processing(rule_1=True, rule_2=True, postfix=True, filename=''):
    train_data = read_math23k_json('./data/math23k/math23k_train.json')
    test_data = read_math23k_json('./data/math23k/math23k_test.json')
    #data_sni=read_data_json(r'data\math23k\sni_DNS.json')
    
    new_train_data = []
    new_test_data = []
    new_valid_data = []
    change_list = []

    '''train'''
    if rule_1:
        rule1_list = rule1_stats(train_data)
    for d in train_data:
        if rule_1:
            if rule_2:
                new_data, change = data_process(d, rule1_list)
            else:
                new_data, change = rule_1_post_data_process(d, rule1_list)
        else:
            if rule_2:
                new_data, change = rule_2_post_data_process(d)
            else:
                new_data, change = post_data_process(d)
        if new_data != None:
            new_train_data.append(new_data)
        if change != None:
            change_list.append(change)
    '''test'''
    if rule_1:
        rule1_list = rule1_stats(test_data)
    for d in test_data:
        if rule_1:
            if rule_2:
                new_data, change = data_process(d, rule1_list)
            else:
                new_data, change = rule_1_post_data_process(d, rule1_list)
        else:
            if rule_2:
                new_data, change = rule_2_post_data_process(d)
            else:
                new_data, change = post_data_process(d)
        if new_data != None:
            new_test_data.append(new_data)
        if change != None:
            change_list.append(change)
    random.shuffle(new_train_data)
    #random.shuffle(new_test_data)
    
    new_valid_data = new_train_data[:1000]
    new_train_data = new_train_data[1000:]
    if filename != '':
        path = "./experiment/change_" + filename + ".json"
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(change_list, f, indent=4)
    return new_train_data, new_valid_data, new_test_data
def test():
    train,valid,test=math23k_data_processing(True,True,True)
    print("1,1,1:\ntrain:{}\ntest:{}".format(len(train),len(test)))
    train,valid,test=math23k_data_processing(True,False,True)
    print("1,0,1:\ntrain:{}\ntest:{}".format(len(train),len(test)))
    train,valid,test=math23k_data_processing(False,True,True)
    print("0,1,1:\ntrain:{}\ntest:{}".format(len(train),len(test)))
    train,valid,test=math23k_data_processing(False,False,True)
    print("0,0,1:\ntrain:{}\ntest:{}".format(len(train),len(test)))
def stat_process():
    print("reading...")
    train_data = read_math23k_json('./data/math23k/math23k_train.json')
    test_data = read_math23k_json('./data/math23k/math23k_test.json')
    #data_sni=read_data_json(r'data\math23k\sni_DNS.json')
    
    train_dict = {"no_norm": [], "1": [], "2": [], "p": [], "1+2": [], "1+p": [], "2+p": [], "1+2+p": []}
    test_dict = {"no_norm": [], "1": [], "2": [], "p": [], "1+2": [], "1+p": [], "2+p": [], "1+2+p": []}
    print("statistics process of train...")
    '''train'''
    rule1_list = rule1_stats(train_data)
    for d in train_data:
        new_data1, change = data_process(d, rule1_list)
        new_data2, change = rule_1_post_data_process(d, rule1_list)
        new_data3, change = rule_2_post_data_process(d)
        new_data4, change = post_data_process(d)
        if new_data4 != None:
            train_dict["no_norm"].append(new_data4['target_template'])
            train_dict["p"].append(new_data4['target_norm_post_template'])
        if new_data2 != None:
            train_dict["1"].append(new_data2['target_template'])
            train_dict["1+p"].append(new_data2['target_norm_post_template'])
        if new_data3 != None:
            train_dict["2"].append(new_data3['target_template'])
            train_dict["2+p"].append(new_data3['target_norm_post_template'])
        if new_data1 != None:
            train_dict["1+2"].append(new_data1['target_template'])
            train_dict["1+2+p"].append(new_data1['target_norm_post_template'])
    print("statistics process of test...")
    '''test'''
    rule1_list = rule1_stats(test_data)
    for d in test_data:
        new_data1, change = data_process(d, rule1_list)
        new_data2, change = rule_1_post_data_process(d, rule1_list)
        new_data3, change = rule_2_post_data_process(d)
        new_data4, change = post_data_process(d)
        if new_data4 != None:
            test_dict["no_norm"].append(new_data4['target_template'])
            test_dict["p"].append(new_data4['target_norm_post_template'])
        if new_data2 != None:
            test_dict["1"].append(new_data2['target_template'])
            test_dict["1+p"].append(new_data2['target_norm_post_template'])
        if new_data3 != None:
            test_dict["2"].append(new_data3['target_template'])
            test_dict["2+p"].append(new_data3['target_norm_post_template'])
        if new_data1 != None:
            test_dict["1+2"].append(new_data1['target_template'])
            test_dict["1+2+p"].append(new_data1['target_norm_post_template'])
    
    return train_dict,test_dict
def Template_Statistics():
    tr, te = stat_process()
    train_template = {"no_norm": [], "1": [], "2": [], "p": [], "1+2": [], "1+p": [], "2+p": [], "1+2+p": []}
    test_template = {"no_norm": [], "1": [], "2": [], "p": [], "1+2": [], "1+p": [], "2+p": [], "1+2+p": []}
    print("template statistics of train...")
    for k, v in tr.items():
        for temp in v:
            if temp not in train_template[k]:
                train_template[k].append(temp)
    for k, v in train_template.items():
        print('train:[{}]:{}'.format(k, len(v)))
    print("template statistics of test...")
    for k, v in te.items():
        for temp in v:
            if temp not in test_template[k]:
                test_template[k].append(temp)
    for k, v in test_template.items():
        print('test:[{}]:{}'.format(k, len(v)))
    print("template statistics of both...")
    for k, v in te.items():
        for temp in v:
            if temp not in train_template[k]:
                train_template[k].append(temp)
    for k, v in train_template.items():
        print('train+test:[{}]:{}'.format(k,len(v)))

def write_json_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    f.close()
def load_processed23k_data():
    path_tr = './data/train23k_processed.json'
    path_te = './data/test23k_processed.json'
    path_va = './data/valid23k_processed.json'
    train = read_data_json(path_tr)
    valid = read_data_json(path_va)
    test = read_data_json(path_te)
    return train, valid, test
def sni_statistics():
    train, valid, test = load_processed23k_data()
    datas = train + valid + test
    right = 0
    wrong = 0
    i=0
    for data in datas:
        template = data['target_template']
        num_list = data['num_list']
        sni_count = 0
        if i == 0:
            print(template)
            print(num_list)
            i += 1
        repetition = {}
        for symbol in template:
            
            if 'temp' in symbol:
                try:
                    repetition[symbol]
                    repetition[symbol] += 1
                except: 
                    sni_count += 1
                    repetition[symbol] = 0
                    
            if 'PI' in symbol:
                try:
                    repetition[symbol]
                    repetition[symbol] += 1
                except: 
                    sni_count += 1
                    repetition[symbol] = 0
            try:
                float(symbol)
                try:
                    repetition[symbol]
                    repetition[symbol] += 1
                except:
                    sni_count += 1
                    repetition[symbol] = 0
            except:
                pass
            
            
        if sni_count == len(num_list):
            right += 1
        else:
            if i < 10:
                i += 1
                
                print(template)
                print(num_list)
            wrong += 1
    print(right)
    print(wrong)


if __name__ == "__main__":
    # train, valid, test = math23k_data_processing(False, False, False)
    # path_tr = './data/train23k_processed.json'
    # path_te = './data/test23k_processed.json'
    # path_va = './data/valid23k_processed.json'
    # write_json_data(train, path_tr)
    # write_json_data(valid, path_va)
    # write_json_data(test, path_te)
    sni_statistics()
    
    