#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import string
import numpy as np

def read_data(train_file_name,test_file_name):
    data={'train_text':None,
          'trian_name':None,
          'name_id':None,
          'test_text':None,
          'test_name':None}                 #data return
    train_text=[]                   #存储bug report中的文本，以string类型
    train_name=[]                   #存储对应bug report的维护者姓名
    test_text=[]                      
    test_name=[]
    #data_list=[]
    #word_num={}
    name_id={}
    ###------------------------------------------------#
    #           将文本文件以列表形式存到data_list
    #           将词-频率对存入word_num
    #           将维护人名字存入name_id
    #------------------------------------------------###
    train_file=open(train_file_name,'r')
    test_file=open(test_file_name,'r')
    #out_file=open('text.txt','w')
    #读取训练集
    i=0
    for line in train_file:
        line='line='+line
        exec line                           #line表示为list类型了
        #data_list.append(line)              #保存原始数据，没用
        if not name_id.has_key(line[2]):
            name_id[line[2]]=i;
            i=i+1
        train_name.append(line[2])
        words=line[5]
        for sentence in line[6]:
            words=words+sentence
        train_text.append(words)
    data['train_text']=train_text
    data['train_name']=train_name
    data['name_id']=name_id
    #读取测试集
    for line in test_file:
        line='line='+line
        exec line
        #data_list.append(line)              #保存原始数据，没用
        test_name.append(line[2])
        words=line[5]
        for sentence in line[6]:
            words=words+sentence
        test_text.append(words)
    data['test_text']=test_text
    data['test_name']=test_name
    train_file.close()
    test_file.close()
    return data
    
#from sklearn.feature_extraction.text import TfidfVectorizer
#此处使用sklearn有问题，sklearn的tfidf不支持idf的重用
#改用老师推荐的gensim
import gensim
from gensim import corpora,models
from collections import defaultdict

def data_vectorizer(data,min_frequency=1):
    
    data_vec={'train_tfidfvec':None,
              'train_namevec':None,
              'test_tfidfvec':None,
              'test_namevec':None}
    #训练集向量化
    '''
    tv=TfidfVectorizer(stop_words='english',
                   lowercase=False,
                   max_features=max_features,
                   dtype=np.float32)

    train_tfidfvector=tv.fit_transform(data['train_text'])
    '''
    frequency=defaultdict(int)
    for text in data['train_text']:
        for token in text:
            frequency[token]+=1
    
    texts=[[token for token in text if frequency[token] > min_frequency]
        for text in data['train_text']]
    
    dictionary=corpora.Dictionary(texts)
    #dictionary.save('eclipse.dict')                 #store to disk
    
    corpus=[dictionary.doc2bow(text) for text in texts]
    #corpora.MmCorpus.serialize('train.mm',corpus)   #store to disk
    
    tfidf_model=models.TfidfModel(corpus)
    corpus_tfidf=tfidf_model[corpus]
    
    train_tfidfvector=gensim.matutils.corpus2dense(corpus_tfidf,num_terms=len(dictionary))
    data_vec['train_tfidfvec']=train_tfidfvector.transpose()
    
    name_dense=np.zeros((len(data['train_name']),),dtype=np.int32)
    i=0
    for x in data['train_name']:
        name_dense[i]=data['name_id'][x]
        i+=1
    num_name = name_dense.shape[0]
    num_classes = len(data['name_id'])
    index_offset = np.arange(num_name) * num_classes
    name_one_hot = np.zeros((num_name, num_classes),dtype=np.float32)
    name_one_hot.flat[index_offset + name_dense.ravel()] = 1
    data_vec['train_namevec']=name_one_hot
    #测试集向量化
    '''
    tv.use_idf=False
    tv.vocabulary=tv.get_feature_names()
    test_tfidfvecter=tv.fit_transform(data['test_text'])
    '''
    corpus=[dictionary.doc2bow(text) for text in data['test_text']]
    #corpora.MmCorpus.serialize('test.mm',corpus)
    
    corpus_tfidf=tfidf_model[corpus]
    
    test_tfidfvector=gensim.matutils.corpus2dense(corpus_tfidf,num_terms=len(dictionary))
    data_vec['test_tfidfvec']=test_tfidfvector.transpose()
    #原本想法是多设一位one-hot编码用来
    #表示新出的人名，问了老师后觉得还是用
    #全0表示新出的人名
    name_dense1=np.zeros((len(data['test_name']),),dtype=np.int32)
    i=0
    offset=[]
    idnum=[]
    for x in data['test_name']:
        if data['name_id'].has_key(x):
            name_dense1[i]=data['name_id'][x]
            offset.append(i)
            idnum.append(data['name_id'][x])
        i+=1
    num_name1=name_dense1.shape[0]
    index_offset1 = np.array(offset) * num_classes
    idnum1=np.array(idnum)
    name_one_hot1 = np.zeros((num_name1, num_classes),dtype=np.float32)
    name_one_hot1.flat[index_offset1 + idnum1.ravel()] = 1
    data_vec['test_namevec']=name_one_hot1

    return data_vec

import tensorflow as tf
class DataSet(object):
    def __init__(self,textvec,namevec,dtype=tf.float32):
        assert textvec.shape[0] == namevec.shape[0], (
            'textvec.shape: %s namevec.shape: %s' % (textvec.shape,
                                               namevec.shape))
        self._num_examples = textvec.shape[0]
        self._textvec = textvec
        self._namevec = namevec
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def textvec(self):
        return self._textvec
        
    @property
    def namevec(self):
        return self._namevec
        
    @property
    def num_examples(self):
        return self._num_examples
        
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._textvec = self._textvec[perm]
            self._namevec = self._namevec[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._textvec[start:end], self._namevec[start:end]
    
def read_data_sets(train_file_name, test_file_name,
                   min_frequency=1,
                   dtype=tf.float32):
    class DataSets(object):
        pass
    data_sets = DataSets()
    
    data=read_data(train_file_name,test_file_name)
    data_vec=data_vectorizer(data,min_frequency)
    
    data_sets.train=DataSet(data_vec['train_tfidfvec'],
                            data_vec['train_namevec'])
    
    data_sets.test=DataSet(data_vec['test_tfidfvec'],
                            data_vec['test_namevec'])
    return data_sets
