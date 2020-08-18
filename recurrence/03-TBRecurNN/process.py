import jieba
import numpy as np
import json
import random
import copy
from .process_tools import *
from ..json_tools import *
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
        return temp_data
    else:
        return None

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
        return temp_data
    else:
        return None

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
        return temp_data
    else:
        return None

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
        return temp_data
    else:
        return None

def math23k_data_processing(rule_1=True, rule_2=True, postfix=True):
    """
    process math23k with rules.
    ---------------------
    optional parameters:
    rule_1   : true or false to control using rule 1 for equation template.
    rule_2   : true or false to control using rule 2 for equation template.
    postfix  : true or false to control using postfix for equation template.
    ---------------------
    return:
    processed train data, valid data and test data
    """
    train_data = read_math23k_json('./data/math23k/math23k_train.json')
    test_data = read_math23k_json('./data/math23k/math23k_test.json')
    
    new_train_data = []
    new_test_data = []
    new_valid_data = []

    '''train'''
    if rule_1:
        rule1_list = rule1_stats(train_data)
    for d in train_data:
        if rule_1:
            if rule_2:
                new_data = data_process(d, rule1_list)
            else:
                new_data = rule_1_post_data_process(d, rule1_list)
        else:
            if rule_2:
                new_data = rule_2_post_data_process(d)
            else:
                new_data = post_data_process(d)
        if new_data != None:
            new_train_data.append(new_data)
    '''test'''
    if rule_1:
        rule1_list = rule1_stats(test_data)
    for d in test_data:
        if rule_1:
            if rule_2:
                new_data = data_process(d, rule1_list)
            else:
                new_data = rule_1_post_data_process(d, rule1_list)
        else:
            if rule_2:
                new_data = rule_2_post_data_process(d)
            else:
                new_data = post_data_process(d)
        if new_data != None:
            new_test_data.append(new_data)

    random.shuffle(new_train_data)
    
    new_valid_data = new_train_data[:1000]
    new_train_data = new_train_data[1000:]
    return new_train_data, new_valid_data, new_test_data

def load_processed23k_data():
    """
    load processed math23k data without rule 1, rule 2 and postfix. 
    """
    path_tr = './data/train23k_processed.json'
    path_te = './data/test23k_processed.json'
    path_va = './data/valid23k_processed.json'
    train = read_data_json(path_tr)
    valid = read_data_json(path_va)
    test = read_data_json(path_te)
    return train, valid, test