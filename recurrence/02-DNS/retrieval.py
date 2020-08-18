from collections import Counter
from gensim import corpora, models, similarities, matutils
import torch

from process import *
from dataload import math23kDataLoader
class Retrieval():
    def __init__(self,dataset,vocab_list,cut_type,cuda_use,theta=0.5):
        super().__init__()
        #self.vocab_list = vocab_list
        self.dataset=self.make_dataset(dataset)
        #self.problem_num = len(self.dataset)
        self.docs = len(self.dataset)
        #self.docu_num = self.count_docu_num()
        self.cut_type = cut_type
        self.theta = theta
        self.cuda_use=cuda_use
        self.dictionary, self.tfidf_model, self.index, self.matrix_2d = self.count_docu_num()
        self.feature_num = len(self.dictionary.keys())
    def make_dataset(self, dataset):
        new_dataset = []
        for batch_data in dataset:
            batch_size=len(batch_data['id'])
            for i in range(batch_size):
                new_data = {}
                new_data['id'] = batch_data['id'][i:i+1]
                new_data['train_var'] = batch_data['train_var'][i:i+1]
                #new_data['sentence_var']=batch_data['train_var'][i].data.tolist()
                new_data['target_var'] = batch_data['target_var'][i: i + 1]
                new_data['text']=batch_data['sentence'][i]
                new_data['equ']=batch_data['equ'][i]
                new_dataset.append(new_data)
        return new_dataset
    
    def count_docu_num(self):
        '''使用gensim构造tfidf模型'''

        '''文本预处理'''
        sentence_list = []
        i = 0
        for data in self.dataset:                                       #dataset为train data
            word_list = [word for word in data['text'].split(' ')]      #分词
            if self.cut_type:                                           #True：将词语分成单个字
                new_word_list = []
                for word in word_list:
                    if 'temp' in word:
                        new_word_list.append(word)
                    elif 'PI' in word:
                        new_word_list.append(word)
                    else:
                        for char in word:
                            new_word_list.append(char)
                word_list = new_word_list
            if i == 0:
                print(word_list)
                #output:
                #['超', '市', '购', '进', '了', '一', '批', '矿', '泉', '水', '，',
                #  '已', '经', '卖', '出', 'temp_a', '，', '还', '剩', 'temp_b', '箱', '，',
                #  '超', '市', '购', '进', '了', '多', '少', '箱', '矿', '泉', '水', '？']
                i += 1
            sentence_list.append(word_list)
        print(len(sentence_list))                                
        #output:22155
        dictionary = corpora.Dictionary(sentence_list)                  #建立字典对象 
        corpus = [dictionary.doc2bow(one) for one in sentence_list]     #构建词袋,得到（word id,num）的词频数据
        print(corpus[0])
        #output:
        #[(0, 1), (1, 1), (2, 1), (3, 2), (4, 1), (5, 1), (6, 1), (7, 1),
        # (8, 1), (9, 1), (10, 2), (11, 1), (12, 2), (13, 2), (14, 2), (15, 2),
        # (16, 1), (17, 2), (18, 2), (19, 1), (20, 2), (21, 3), (22, 1)]
        '''模型'''
        tfidf_model = models.TfidfModel(corpus)                         #建立模型
        tfidf_value_list = tfidf_model[corpus]                          #计算train data每个data的tfidf值，得到元素(id,value)的编号矩阵
        print(tfidf_value_list[0])
        #output:
        #[(1, 0.000836818408456615), (2, 0.022290065319798102),
        # (3, 0.08910755152596495)......(22, 0.012364676522983354)]
        
        feature_num = len(dictionary.keys())                            #词的数量
        #a = matutils.corpus2dense(p_list, feature_num)
        list_2d = []
        for one in tfidf_value_list:                                    #将编号矩阵转换为长度为feature_num的向量
                                                                        #所有train data的向量构成2维的矩阵(22155, 2494)
            list_1d = self.VecToSparseMatrix(id_vector=one, matrix_len=feature_num)
            list_2d.append(list_1d)
        print(np.array(list_2d).shape)
        #output:(22155, 2494)
        return dictionary, tfidf_model, corpus, list_2d
    
    def VecToSparseMatrix(self, id_vector, matrix_len=None):
        if matrix_len == None:
            sparsematrix = [0.] * self.feature_num
        else:
            sparsematrix = [0.] * matrix_len
        for i, v in id_vector:
            sparsematrix[i] = v
        return sparsematrix

    def max_similarity(self, p_test):
        p_test = [word for word in p_test.split(' ')]
        if self.cut_type:
            new_p_test = []
            for word in p_test:
                if 'temp' in word:
                    new_p_test.append(word)
                elif 'PI' in word:
                    new_p_test.append(word)
                else:
                    for char in word:
                        new_p_test.append(char)
            p_test = new_p_test
        p_test = self.dictionary.doc2bow(p_test)
        sim = self.index[self.tfidf_model[p_test]]
        max_sim = np.max(sim)
        idx = np.argmax(sim)
        return max_sim, self.dataset[idx]['target_var'],self.dataset[idx]['text']
    
    def max_js(self, p_test,p_id=None):
        max_similarity = 0
        target_temp=None
        test_vec = self.prob2vec(p_test.data.tolist()[0])
        for data in self.dataset:
            if p_id == data['id'][0]:
                continue
            target_vec = self.prob2vec(data['sentence_var'])
            similarity = self.jaccard_similarity(test_vec, target_vec)
            if similarity > max_similarity:
                max_similarity = similarity
                target_temp=data['target_var']
        return similarity, target_temp
    
    def js(self, p_test):
        #input:
        # 1000个test中的其中1个
        '''计算相似度'''
        print(p_test)
        #output:
        #'在 一 正方形 花池 的 temp_a 周栽 了 temp_b 棵 柳树 ， 每 两棵 柳树 之间 的 间隔 是 temp_c 米 ， 这个 正方形 的 周长 = 多少 米 ？'
        p_test = [word for word in p_test.split(' ')]
        if self.cut_type:
            new_p_test = []
            for word in p_test:
                if 'temp' in word:
                    new_p_test.append(word)
                elif 'PI' in word:
                    new_p_test.append(word)
                else:
                    for char in word:
                        new_p_test.append(char)
            p_test = new_p_test
        print(p_test)
        #output:
        #['在', '一', '正', '方', '形', '花', '池', '的', 'temp_a', '周', '栽', '了', 'temp_b', '棵', '柳', '树', '，',
        # '每', '两', '棵', '柳', '树', '之', '间', '的', '间', '隔', '是', 'temp_c', '米', '，
        # ', '这', '个', '正', '方', '形', '的', '周', '长', '=', '多', '少', '米', '？']
        p_test = self.dictionary.doc2bow(p_test)
        print(p_test)
        #output:
        #[(0, 1), (1, 1), (2, 1), (3, 1), (7, 1), (8, 1), (21, 2), (22, 1),
        # (23, 1), (35, 1), (37, 3), (39, 1), (41, 1), (77, 2), (85, 2),
        # (90, 1), (94, 1), (161, 1), (164, 2), (200, 1), (209, 2), (250, 2),
        # (297, 1), (308, 1), (313, 1), (344, 2), (378, 2), (519, 2),
        # (889, 1), (966, 1), (1009, 2), (1050, 1)]
        p_test = self.tfidf_model[p_test]                                   #计算test的tfidf值
        print(p_test)
        #output:
        #[(1, 0.0009350331184527052), (2, 0.02490617925688936),
        # (3, 0.04978291044927925)......(1050, 0.21435616918212957)]
        test_matrix = self.VecToSparseMatrix(p_test)                        #转为长度为feature_num的向量 （2494）
        test_matrix_2d = [test_matrix] * self.docs                          #test的向量构造成2维矩阵(22155, 2494)，
                                                                            #self.docs==num of train data==22155
                                                                            #与train data的矩阵大小相对应，方便与每一个train计算交集和并集
        if self.cuda_use:
            matrix_3d = torch.tensor([self.matrix_2d, test_matrix_2d])      #拼接train和test，(2, 22155, 2494)
            #a=torch.min(matrix_3d, dim=0)
            #b=torch.max(matrix_3d, dim=0)
            inter = torch.sum(torch.min(matrix_3d, dim=0).values,dim=1)     #计算交集(2, 22155, 2494)-->(22155, 2494)-->(22155)
            union = torch.sum(torch.max(matrix_3d, dim=0).values,dim=1)     #计算并集(2, 22155, 2494)-->(22155, 2494)-->(22155)
            '''similarity'''
            sim_matrix = inter / union                                      #计算相似度(22155)
            max_sim = torch.max(sim_matrix)                                 #最大相似度，0.4964
            idx = torch.argmax(sim_matrix)                                  #最大相似度的索引，19773
            print(max_sim)
            print(idx)
            #output:
            #tensor(0.4964, dtype=torch.float64)
            # tensor(19773)
            #return self.dataset[idx]['target_var']                          #返回方程模板
            return self.sample_top5(sim_matrix)
        else:
            matrix_3d = np.array([self.matrix_2d, test_matrix_2d])
            inter = np.min(matrix_3d, axis=0).sum(axis=1)
            union = np.max(matrix_3d, axis=0).sum(axis=1)
            '''similarity'''
            sim_matrix = inter / union
            max_sim = np.max(sim_matrix)
            print(max_sim)
            idx = np.argmax(sim_matrix)
            print(idx)
        
        return max_sim, self.dataset[idx]['target_var'], self.dataset[idx]['text']
    def sample_top5(self, sim):
        sim, ind = torch.sort(sim, descending=True)
        sim5 = sim.data.tolist()[0:5]
        ind5 = ind.data.tolist()[0:5]
        datas=[]
        for i,idx in enumerate(ind5):
            data={}
            data['text']=self.dataset[idx]['text']
            data['equ'] = self.dataset[idx]['equ']
            data['similarity'] = sim5[i]
            datas.append(data)

        return datas
    def jaccard_similarity(self, p_one, p_two):
        p_one = [w[1] for w in p_one]
        p_two = [w[1] for w in p_two]
        p_one = np.array(p_one)
        p_two = np.array(p_two)
        inter=np.intersect1d(p_one,p_two)
        #inter=[w for w in p_one if w in p_two]
        return len(inter)/(len(p_one)+len(p_two)-len(inter))
    def prob2vec(self, text_list):
        
        freq_dict = Counter(text_list)
        text_len=len(text_list)
        vector = []
        for word in text_list:
            try:
                weight = freq_dict[word] / text_len * self.problem_num / self.docu_num[word]
            except:
                weight=float('inf')
            vector.append(weight)
        return vector


if __name__ == "__main__":
    print(1)
