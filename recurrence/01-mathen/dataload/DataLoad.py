import json
import numpy as np
from gensim.models import word2vec
import torch
from torch.autograd import Variable
class math23kDataLoader():
    def __init__(self,train_data,test_data,valid_data):
        super().__init__()
        # self.train_data=self.read_data_json("./data/math23k/train23k_processed.json")
        # self.test_data=self.read_data_json("./data/math23k/test23k_processed.json")
        # self.valid_data=self.read_data_json("./data/math23k/valid23k_processed.json")
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data

        self.emb_vectors,self.vocab_list,self.decode_classes_list = self.preprocess_and_word2vec(emb_dim=128)
        self.vocab_2_ind = self.word2ind(self.vocab_list)
        self.decode_classes_2_ind = self.word2ind(self.decode_classes_list)

        self.vocab_len = len(self.vocab_list)
        self.decode_classes_len = len(self.decode_classes_list)
    def read_data_json(self,filename):
        f=open(filename, 'r')
        return json.load(f)
    def preprocess_and_word2vec(self, emb_dim):
        new_data = {}
        sentences = []
        equ_dict = {}
        for elem in self.train_data:
            sentence = elem['text'].strip().split(' ')
            
            equation = elem['target_template'][2:]#.strip().split(' ')
            
            for equ_e in equation:
                if equ_e not in equ_dict:
                    equ_dict[equ_e] = 1
            sentences.append(sentence)
            for elem in sentence:
                new_data[elem] = new_data.get(elem, 0) + 1
        
        model = word2vec.Word2Vec(sentences, size=emb_dim, min_count=1)

        token_list = ['PAD_token', 'UNK_token', 'END_token']
        ext_list = ['PAD_token', 'END_token']
        emb_vectors = []
        emb_vectors.append(np.zeros((emb_dim)))
        emb_vectors.append(np.random.rand((emb_dim))/1000.0)
        emb_vectors.append(np.random.rand((emb_dim))/1000.0)

        for k, v in new_data.items(): 
            token_list.append(k)
            emb_vectors.append(np.array(model.wv[k]))

        for equ_k in equ_dict.keys():
            ext_list.append(equ_k)
        print ("encode_len:", len(token_list), "decode_len:", len(ext_list))
        print ("de:",ext_list)
        for elem in ext_list:
            if elem not in token_list:
                token_list.append(elem)
                emb_vectors.append(np.random.rand((emb_dim))/1000.0)
        emb_vectors = np.array(emb_vectors)
        
        return emb_vectors, token_list, ext_list
    def _convert_f_e_2_d_sybmbol(self, target_variable):
        new_variable = []
        batch,colums = target_variable.size()
        for i in range(batch):
            tmp = []
            for j in range(colums):
                #idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].data[0]]]
                idx = self.decode_classes_2_ind[self.vocab_list[target_variable[i][j].item()]]
                tmp.append(idx)
            new_variable.append(tmp)
        return Variable(torch.LongTensor(np.array(new_variable)))
    def word2ind(self,wordlist):
        word2ind = {}
        for index,item in enumerate(wordlist):
            word2ind[item] = index
        return word2ind
    def loading(self,data_type,batch_size,postfix):
        '''
        data_type:"train"/"test"/"valid"
        '''
        if data_type == "train":
            data = self.train_data
        elif data_type == "valid":
            data = self.valid_data
        else:
            data = self.test_data
        max_data_len=len(data)
        batch = int(max_data_len/batch_size)+1
        loaded_data = []
        for i in range(batch):
            if max_data_len >= (i + 1) * batch_size:
                batch_data = data[i * batch_size : (i + 1) * batch_size]
            else:
                batch_data = data[i * batch_size :max_data_len]
            '''text'''
            batch_var = []
            batch_len = []
            batch_text = []
            for text in batch_data:
                vec = []
                for j in text['text'].split(' '):
                    if j not in self.vocab_2_ind:
                        vec.append(self.vocab_2_ind['UNK_token'])
                    else:
                        vec.append(int(self.vocab_2_ind[j]))
                sentence = text['text']
                batch_text.append(sentence)
                batch_len.append(len(vec))
                batch_var.append(vec)
            max_len = max(batch_len)
            batch_var = [vec + [self.vocab_2_ind['PAD_token']] * (max_len - len(vec)) for vec in batch_var]
            batch_var = torch.LongTensor(batch_var)
            '''template'''
            batch_tem_len = []
            batch_tem_var = []
            for templete in batch_data:
                vec = []
                if postfix:
                    for j in templete['target_norm_post_template'][2:]:
                        if j not in self.vocab_2_ind:
                            vec.append(self.vocab_2_ind['UNK_token'])
                        else:
                            vec.append(self.vocab_2_ind[j])
                else:
                    for j in templete['target_template'][2:]:
                        if j not in self.vocab_2_ind:
                            vec.append(self.vocab_2_ind['UNK_token'])
                        else:
                            vec.append(self.vocab_2_ind[j])
                vec.append(self.vocab_2_ind['END_token'])
                batch_tem_len.append(len(vec))
                batch_tem_var.append(vec)
            max_len = max(batch_tem_len)
            batch_tem_var = [vec + [self.vocab_2_ind['PAD_token']] * (max_len - len(vec)) for vec in batch_tem_var]
            batch_tem_var = torch.LongTensor(batch_tem_var)
            '''id num_list answer'''
            batch_id = []
            batch_num_list = []
            batch_solution = []
            #batch_truth_equ = []
            for d in batch_data:
                batch_id.append(d['id'])
                batch_num_list.append(d['num_list'])
                batch_solution.append(d['answer'])
            loaded_data.append({'train_var':batch_var,'train_len':batch_len,'target_var':batch_tem_var,
                                'target_len':batch_tem_len,'sentence':batch_text,'id':batch_id,
                                'num_list':batch_num_list,'solution':batch_solution})
        return loaded_data
#math=math23kDataLoader()
#print(math.decode_classes_2_ind)
