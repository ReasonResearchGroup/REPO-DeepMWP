import jieba
import numpy as np
import json
def read_math23k_json(filename):
    data_list = []
    with open(filename, 'r',encoding="utf-8") as f:
        count = 0
        string = ''
        for line in f:
            count += 1
            string += line
            if count % 7 == 0:
                data_list.append(json.loads(string))
                string = ''
    return data_list
def data_process(text):
    text=jieba.cut(text,cut_all=False)
    return text
if __name__ == "__main__":
    path=r'data\math23k\math23k_train.json'
    data=read_math23k_json(path)
    for d in data:
        text=data_process(d['original_text'])
        print(text)
        break