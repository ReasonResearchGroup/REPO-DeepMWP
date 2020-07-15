import jieba
import numpy as np
import json
def read_data_json(filename):
    #with open(filename, 'r') as f:
    f=open(filename, 'r')
    return json.load(f)
def read_math23k_json(filename):
    data_list = []
    #with open(filename, 'r',encoding="utf-8") as f:
    f=open(filename, 'r',encoding="utf-8")
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
def mask_num(seg_text_list, equ_str):
    origin_equ_str = equ_str[:]

    alphas = 'abcdefghijklmnopqrstuvwxyz'
    num_list  = []
    mask_seg_text = []
    count = 0 
    for word in seg_text_list:
        if word == '':
            continue
        if is_number(word):
            mask_seg_text.append("temp_"+alphas[count])
            if '%' in word:
                mask_seg_text.append('%')
            #elif 'm' in word.lower() or 'g' in word.lower() or 'd' in word.lower():
            elif len(set(alphas)&set(word.lower()))>0:
                num, unit = split_num_and_unit(word)
                mask_seg_text.append(unit)
                word = num
                
            num_list.append(word)
            count += 1
        else:
            mask_seg_text.append(word)
    mask_equ_list = []
    s_n = sorted([(w,i) for i,w in enumerate(num_list)], key=lambda x: len(str(x[0])), reverse=True)
    if '3.14%' not in equ_str and '3.1416' not in equ_str:
        equ_str = equ_str.replace('3.14', '&PI&', 15)
    new_equ_str = ''
    #print (s_n)
    #print (equ_str)
    for num, idx in s_n:
        #num = num_list[idx]
        equ_str = equ_str.replace(num, '&temp_'+alphas[idx]+'&', 15)
        #if 
    #print (equ_str)
        
        
    equ_list = []
    num_set = ['0','1','2','3','4','5','6','7','8','9','%', '.']
    for elem in equ_str.split('&'):
        if 'temp' in elem or 'PI' in elem:
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
    return mask_seg_text, num_list, equ_list#, reverse_equ_list

def mask_text(seg_text_list):
    alpha="abcdefghijklmnopqrstuvwxyz"
    count=0
    num_dict={}
    new_text=[]
    for word in seg_text_list:
        unit=''
        if is_number(word):
            if len(set(alpha)&set(word.lower()))>0:
                num, unit = split_num_and_unit(word)
                word = num
            try:
                num_dict[word]
            except: 
                num_dict[word]="temp_"+alpha[count]
                count+=1
            
            #count+=1
            new_text.append(num_dict[word])
            
            #if "%" in word:
            #   new_text.append("%")
            if unit !='':
                new_text.append(unit)
        else:
            new_text.append(word)
    return num_dict,new_text
def mask_equ(equ_str,num_dict):
    if '3.14%' not in equ_str and '3.1416' not in equ_str:
        equ_str = equ_str.replace('3.14', '&PI&', 15)
    num_dict=sorted(num_dict.items(),key=lambda x:len(x[0]),reverse=True)
    for k,v in num_dict:
        equ_str=equ_str.replace(k,"&{}&".format(v),15)
    equ_list=[]
    num_set=['0','1','2','3','4','5','6','7','8','9','%', '.']
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
            num_,_ = split_num_and_unit(num)
            new_num_list.append(float(num_))
    return new_num_list

def norm_equ(equ_list):
    '''
    only for post
    '''
    i = 0
    new_equ_list = []
    #print (equ_list)
    while i < len(equ_list):
        #if i-1>=0 and equ_list[i-1] not in ['/','-'] and 'temp' in equ_list[i] and (i+5) < len(equ_list) and equ_list[i+5] not in ['/','-'] 'temp' in equ_list[i+2] and equ_list[i+1] == '+' and equ_list[i+3] == '+' and 'temp' in equ_list[i+4]:
        if 'temp' in equ_list[i] and (i+4) < len(equ_list) and 'temp' in equ_list[i+2] and equ_list[i+1] == '+' and equ_list[i+3] == '+' and 'temp' in equ_list[i+4]:
            if i-1>=0 and equ_list[i-1] in ['/','-', '*']:
                new_equ_list.append(equ_list[i])
                i+=1
                continue
            if i+5< len(equ_list)  and equ_list[i+5] in ['/','-','*']:
                new_equ_list.append(equ_list[i])
                i+=1
                continue  
            temp = [equ_list[i], equ_list[i+2], equ_list[i+4]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['+']+sort_temp[1:2]+['+']+sort_temp[2:3]
            new_equ_list += new_temp
            i += 5
        #elif 'temp' in equ_list[i] and (i+5) < len(equ_list) and equ_list[i+5] not in ['/','-'] and 'temp' in equ_list[i+2] and equ_list[i+1] == '+' and equ_list[i+3] == '+' and 'temp' in equ_list[i+4]:
        elif 'temp' in equ_list[i] and (i+4) < len(equ_list) and 'temp' in equ_list[i+2] and equ_list[i+1] == '+' and equ_list[i+3] == '+' and 'temp' in equ_list[i+4]:
            if i-1>=0 and equ_list[i-1] in ['/','-']:
                new_equ_list.append(equ_list[i])
                i+=1
                continue
            if i+5< len(equ_list)  and equ_list[i+5] in ['/','-']:
                new_equ_list.append(equ_list[i])
                i+=1
                continue  
            temp = [equ_list[i], equ_list[i+2], equ_list[i+4]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]+['*']+sort_temp[2:3]
            new_equ_list += new_temp
            i += 5
        #elif 'temp' in equ_list[i] and (i+5) < len(equ_list) and equ_list[i+5] not in ['/','-'] and 'temp' in equ_list[i+2] and equ_list[i+1] == '*' and equ_list[i+3] == '*' and 'temp' in equ_list[i+4]:
        elif (i+2) < len(equ_list) and 'temp' in equ_list[i]  and equ_list[i+1] == '+' 'temp' in equ_list[i+2] :
            #print (equ_list[i:i+3])
            
            if i-1>=0 and equ_list[i-1] in ['/','-', '*']:
                new_equ_list.append(equ_list[i])
                i+=1
                continue
            if i+3< len(equ_list)  and equ_list[i+3] in ['/','-', '*']:
                new_equ_list.append(equ_list[i])
                i+=1
                continue  
            temp = [equ_list[i], equ_list[i+2]]
            #print (temp)
            sort_temp = sorted(temp)
            #print (sort_temp)
            #print ()
            new_temp = sort_temp[0:1]+['+']+sort_temp[1:2]
            new_equ_list += new_temp
            i += 3
        elif 'temp' in equ_list[i] and (i+2) < len(equ_list) and 'temp' in equ_list[i+2]  and equ_list[i+1] == '+' and 'temp' in equ_list[i+2] :
            if i-1>=0 and equ_list[i-1] in ['/','-']:
                new_equ_list.append(equ_list[i])
                i+=1
                continue
            if i+3< len(equ_list)  and equ_list[i+3] in ['/','-']:
                new_equ_list.append(equ_list[i])
                i+=1
                continue  
            temp = [equ_list[i], equ_list[i+2]]
            #print (temp)
            sort_temp = sorted(temp)
            #print (sort_temp)
            #print ()
            new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]
            new_equ_list += new_temp
            i += 3
        else:
            new_equ_list.append(equ_list[i])
            i+=1
    
    #print (new_equ_list)
    #print ('----')
    return new_equ_list[:]
def norm_equation(equ_list):
    new_list=[]
    i=0
    while i<len(equ_list):
        if (i+4)<len(equ_list) and 'temp' in equ_list[i] and '+' in equ_list[i+1] and 'temp' in equ_list[i+2] and '+' in equ_list[i+3] and 'temp' in equ_list[i+4]:
            if i-1>=0 and equ_list[i-1] in ['/','-', '*']:
                new_list.append(equ_list[i])
                i+=1
                continue
            if i+5< len(equ_list)  and equ_list[i+5] in ['/','-','*']:
                new_list.append(equ_list[i])
                i+=1
                continue  
            temp = [equ_list[i], equ_list[i+2], equ_list[i+4]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['+']+sort_temp[1:2]+['+']+sort_temp[2:3]
            new_list += new_temp
            i += 5
        elif (i+4)<len(equ_list) and 'temp' in equ_list[i] and '*' in equ_list[i+1] and 'temp' in equ_list[i+2] and '*' in equ_list[i+3] and 'temp' in equ_list[i+4]:
            if i-1>=0 and equ_list[i-1] in ['/','-']:
                new_list.append(equ_list[i])
                i+=1
                continue
            if i+5< len(equ_list)  and equ_list[i+5] in ['/','-']:
                new_list.append(equ_list[i])
                i+=1
                continue  
            temp = [equ_list[i], equ_list[i+2], equ_list[i+4]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]+['*']+sort_temp[2:3]
            new_list += new_temp
            i += 5
        elif (i+2)<len(equ_list) and 'temp' in equ_list[i] and '+' in equ_list[i+1] and 'temp' in equ_list[i+2]:
            if i-1>=0 and equ_list[i-1] in ['/','-', '*']:
                new_list.append(equ_list[i])
                i+=1
                continue
            if i+3< len(equ_list)  and equ_list[i+3] in ['/','-', '*']:
                new_list.append(equ_list[i])
                i+=1
                continue  
            temp = [equ_list[i], equ_list[i+2]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['+']+sort_temp[1:2]
            new_list += new_temp
            i += 3
        elif (i+2)<len(equ_list) and 'temp' in equ_list[i] and '*' in equ_list[i+1] and 'temp' in equ_list[i+2]:
            if i-1>=0 and equ_list[i-1] in ['/','-']:
                new_list.append(equ_list[i])
                i+=1
                continue
            if i+3< len(equ_list)  and equ_list[i+3] in ['/','-']:
                new_list.append(equ_list[i])
                i+=1
                continue  
            temp = [equ_list[i], equ_list[i+2]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]
            new_list += new_temp
            i += 3
        else:
            new_list.append(equ_list[i])
            i+=1
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
def inverse(equ_list,num_dict):
    num_dict={v:k for k,v in num_dict.items()}

    new_equ_list=[]
    for elem in equ_list:
        if "temp" in elem:
            new_equ_list.append(num_dict[elem])
        elif "PI" in elem:
            new_equ_list.append("3.14")
        else:
            new_equ_list.append(elem)
    return new_equ_list
def list2str(equ_list):
    equ_str=''
    for elem in equ_list:
        """ if 'temp' in elem or 'PI' in elem:
            equ_str+="&{}&".format(elem)
        else: """
        equ_str+=elem
    return equ_str

def post_solver(post_equ):
    stack = []
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        if elem not in op_list:
            op_v = elem
            #if '%' in op_v:
            #    op_v = float(op_v[:-1])/100.0
            stack.append(str(op_v))
        elif elem in op_list:
            op_v_1 = stack.pop()
            op_v_1 = float(op_v_1)
            if stack==[]:
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
        i+=1
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
        if str(word)[0]=='(' and str(word)[-1]==')':
            return ans_num_joint(word)
        if str(word)[0] != '(' and str(word)[-1]==')':
            return ans_decimal_exception(word)
    return -float('inf')
def data_process(data):
    
    temp_data=data
    # {"id":"6",
    # "original_text":"10.6-0.4与5.5的积，所得的差除以2.1，商=？",
    # "segmented_text":"10.6 - 0.4 与 5.5 的 积 ， 所得 的 差 除以 2.1 ， 商 = ？",
    # "equation":"x=(10.6-0.4*5.5)/2.1",
    # "ans":"4"}
    text=data['original_text']
    equ_str=data["equation"]
    '''word cut'''
    text=jieba.cut(text,cut_all=False)
    origin_text = ' '.join(text)
    
    word_list=origin_text.split(' ')
    word_list=joint_number(word_list)
    
    '''mask'''
    num_dict,new_text=mask_text(word_list)
    equ_list=mask_equ(equ_str,num_dict)
    
    '''num_list'''
    num_list=list(num_dict.keys())
    num_list=num_list_processed(num_list)
    
    if '千' in equ_list:
        equ_list = equ_list[:equ_list.index('千')]
    
    '''rule 2'''
    rule_2=norm_equation(equ_list)
    
    '''rule 2 + postfix'''
    rule_2_post=postfix_equation(rule_2)
    
    '''answer'''
    post_equ_list=inverse_temp_to_num(rule_2_post,num_list)
    ans=post_solver(post_equ_list[2:])
    
    if abs(float(ans) - float(ans_process(data['ans']))) < 1e-4:
        temp_data['new_split'] =origin_text
        temp_data['text']=' '.join(new_text)
        temp_data["target_norm_post_template"]=rule_2_post
        temp_data["target_template"]=rule_2
        temp_data["num_list"]=num_list
        temp_data["answer"]=float(ans_process(data['ans']))
        return temp_data
    else:
        return None
def math23k_data_process():
    train_data=read_math23k_json(r'data\math23k\math23k_train.json')
    test_data=read_math23k_json(r'data\math23k\math23k_test.json')
    data_sni=read_data_json(r'data\math23k\sni_DNS.json')
    
    new_train_data=[]
    new_test_data=[]
    new_valid_data=[]
    for d in train_data:
        new_data=data_process(d)
        if new_data:
            new_train_data.append(new_data)
    for d in test_data:
        new_data=data_process(d)
        if new_data:
            new_test_data.append(new_data)
    
    new_valid_data=new_train_data[:1000]
    new_train_data=new_train_data[1000:]
    return new_train_data,new_valid_data,new_test_data

if __name__ == "__main__":
    
    train,valid,test=math23k_data_process()
    print(len(train))
    print(len(valid))
    print(len(test))
    template=[]
    template_rule2=[]
    template_post=[]
    template_rule2_post=[]
    
        
        
    """ if equ_str in template:
        continue
    else:
        template.append(equ_str)
    if rule_2 in template_rule2:
        continue
    else:
        template_rule2.append(rule_2)
    if postfix in template_post:
        continue
    else:
        template_post.append(rule_2)
    if rule_2_post in template_rule2_post:
        continue
    else:
        template_rule2_post.append(rule_2) """
    
    print("模板数量：")
    print("before normalization:",len(template))
    print("rule_2:",len(template_rule2))
    print("postfix:",len(template_post))
    print("rule2+postfix:",len(template_rule2_post))
    
    #for i in template_rule2_post:
    #    print(i)
        #equ_text=d["equation"]
        #print(text)
        # word_list=text.split(' ')
        # word_list=joint_number(word_list)
        # #print(i)
        # mask_seg_text, num_list, equ_list=mask_num(word_list,equ_text)
        # num_list=num_list_processed(num_list)
        
        # num_dict,new=mask_text(word_list)
        # equ_list2=mask_equ(equ_text,num_dict)
        # num_list2=list(num_dict.keys())
        # num_list2=num_list_processed(num_list2)
        # print(num_list)
        # print(num_list2)
        #print(equ_list)
        #print(equ_list2)
        #break