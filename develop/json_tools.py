import jieba
import numpy as np
import json
import random
import copy
def read_data_json(filename):
    """
    read data from json file as list or dict object
    """
    f = open(filename, 'r')
    return json.load(f)
def read_math23k_json(filename):
    """
    specially used to read data of math23k file
    """
    data_list = []
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
def write_json_data(data, filename):
    """
    write data to a json file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    f.close()