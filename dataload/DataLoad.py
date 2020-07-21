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
        self.train_data=train_data
        self.test_data=test_data
        self.valid_data=valid_data

        self.emb_vectors,self.vocab_list,self.decode_classes_list = self.preprocess_and_word2vec(emb_dim=128)
        self.vocab_2_ind = self.word2ind(self.vocab_list)
        self.decode_classes_2_ind = self.word2ind(self.decode_classes_list)

        self.vocab_len=len(self.vocab_list)
        self.decode_classes_len=len(self.decode_classes_list)
    def train2vec(self,batch_size,postfix):
        var=[]
        lenth=[]
        target_var=[]
        target_lenth=[]
        _id=[]
        num_list=[]
        solution=[]
        sen=[]    
        batch=int(len(self.train_data)/batch_size)
        for i in range(batch):
            batch_data=self.train_data[i*batch_size:(i+1)*batch_size]
            batch_var=[]
            batch_len=[]
            batch_text=[]
            for text in batch_data:
                #vec=[self.vocab_2_ind['PAD_token']]
                vec=[]
                for j in text['text'].split(' '):
                    if j not in self.vocab_2_ind:
                        vec.append(self.vocab_2_ind['UNK_token'])
                    else:
                        vec.append(int(self.vocab_2_ind[j]))
                sentence=text['text']
                batch_text.append(sentence)
                batch_len.append(len(vec))
                batch_var.append(vec)
            max_len=max(batch_len)
            batch_var=[vec+[self.vocab_2_ind['PAD_token']]*(max_len-len(vec))for vec in batch_var]
            #batch_var=batch_var+[self.vocab_2_ind['PAD_token']]*(max_len-batch_len)
            batch_var=torch.LongTensor(batch_var)
            lenth.append(batch_len)
            var.append(batch_var)
            sen.append(batch_text)
        for i in range(batch):
            batch_data=self.train_data[i*batch_size:(i+1)*batch_size]
            batch_var=[]
            batch_len=[]
            for templete in batch_data:
                vec=[]
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
                batch_len.append(len(vec))
                batch_var.append(vec)
            max_len=max(batch_len)
            batch_var=[vec+[self.vocab_2_ind['PAD_token']]*(max_len-len(vec))for vec in batch_var]
            batch_var=torch.LongTensor(batch_var)
            target_lenth.append(batch_len)
            target_var.append(batch_var)
        for i in range(batch):
            batch_data=self.train_data[i*batch_size:(i+1)*batch_size]
            batch_id=[]
            batch_num_list=[]
            batch_solution=[]
            for data in batch_data:
                batch_id.append(data['id'])
                batch_num_list.append(data['num_list'])
                batch_solution.append(data['answer'])
            _id.append(batch_id)
            num_list.append(batch_num_list)
            solution.append(batch_solution)

        return {'train_var':var,'train_len':lenth,'target_var':target_var,'target_len':target_lenth,
        'sentence':sen,'id':_id,'num_list':num_list,'solution':solution}
    def test2vec(self,batch_size,postfix):
        var=[]
        lenth=[]
        target_var=[]
        target_lenth=[]
        _id=[]
        num_list=[]
        solution=[]
        sen=[]    
        batch=int(len(self.test_data)/batch_size)
        for i in range(batch):
            batch_data=self.test_data[i*batch_size:(i+1)*batch_size]
            batch_var=[]
            batch_len=[]
            batch_text=[]
            for text in batch_data:
                #vec=[self.vocab_2_ind['PAD_token']]
                vec=[]
                for j in text['text'].split(' '):
                    if j not in self.vocab_2_ind:
                        vec.append(self.vocab_2_ind['UNK_token'])
                    else:
                        vec.append(int(self.vocab_2_ind[j]))
                sentence=text['text']
                batch_text.append(sentence)
                batch_len.append(len(vec))
                batch_var.append(vec)
            max_len=max(batch_len)
            batch_var=[vec+[self.vocab_2_ind['PAD_token']]*(max_len-len(vec))for vec in batch_var]
            #batch_var=batch_var+[self.vocab_2_ind['PAD_token']]*(max_len-batch_len)
            batch_var=torch.LongTensor(batch_var)
            lenth.append(batch_len)
            var.append(batch_var)
            sen.append(batch_text)
        for i in range(batch):
            batch_data=self.test_data[i*batch_size:(i+1)*batch_size]
            batch_var=[]
            batch_len=[]
            for templete in batch_data:
                vec=[]
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
                batch_len.append(len(vec))
                batch_var.append(vec)
            max_len=max(batch_len)
            batch_var=[vec+[self.vocab_2_ind['PAD_token']]*(max_len-len(vec))for vec in batch_var]
            batch_var=torch.LongTensor(batch_var)
            target_lenth.append(batch_len)
            target_var.append(batch_var)
        for i in range(batch):
            batch_data=self.test_data[i*batch_size:(i+1)*batch_size]
            batch_id=[]
            batch_num_list=[]
            batch_solution=[]
            for data in batch_data:
                batch_id.append(data['id'])
                batch_num_list.append(data['num_list'])
                batch_solution.append(data['answer'])
            _id.append(batch_id)
            num_list.append(batch_num_list)
            solution.append(batch_solution)

        return {'train_var':var,'train_len':lenth,'target_var':target_var,'target_len':target_lenth,
        'sentence':sen,'id':_id,'num_list':num_list,'solution':solution}
    def read_data_json(self,filename):
        f=open(filename, 'r')
        return json.load(f)
    def preprocess_and_word2vec(self, emb_dim):
        new_data ={}
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
        word2ind={}
        for index,item in enumerate(wordlist):
            word2ind[item]=index
        return word2ind

#math=math23kDataLoader()
#print(math.decode_classes_2_ind)
