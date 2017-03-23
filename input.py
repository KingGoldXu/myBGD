# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import string
import numpy as np

""" file struct [id,time,person,version,model,[object],[describe for detail],[],[],[]]"""
""" decode转化为Unicode，encode转化为str """

word_per_br=100                 #每个bug report中提取的词的长度
word_size=5000                  #学习中词的总个数  
def read_data(train_file_name,test_file_name):
    data={'train_text':None,        #shape:[num_bug_report,word_per_br],type:unicode
          'trian_name':None,        #shape:[num_bug_report]
          'name_id':None,
          'word_id':None,
          'test_text':None,
          'test_name':None,}                 #data return
    train_text=[]                   #存储bug report中的文本，以string类型
    train_name=[]                   #存储对应bug report的维护者姓名
    test_text=[]                      
    test_name=[]
    #data_list=[]
    word_num={}                     #存储词和对应词频，选出最高word_size个词
    name_id={}                      #存储人名对应的id
    word_id={}                      #存储词对应的id
    
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
        words=words[:word_per_br]
        train_text.append(words)
    data['train_text']=train_text
    data['train_name']=train_name
    data['name_id']=name_id
    #print(i)
    #提取词频最高的word_size个词
    for x in train_text:
        for y in x:
            if word_num.has_key(y):
                word_num[y]=word_num[y]+1
            else:
                word_num[y]=1
    sorted_word_num=sorted(word_num.iteritems(),key=lambda t:t[1],reverse=True)
    i=0
    for x in sorted_word_num:
        if i<word_size:
            word_id[x[0]]=i
            i=i+1
    data['word_id']=word_id
    #读取测试集
    for line in test_file:
        line='line='+line
        exec line
        #data_list.append(line)              #保存原始数据，没用
        test_name.append(line[2])
        words=line[5]
        for sentence in line[6]:
            words=words+sentence
        words=words[:word_per_br]
        test_text.append(words)
    data['test_text']=test_text
    data['test_name']=test_name
    train_file.close()
    test_file.close()
    return data

def data_digital(data):
    digit_data={'train_text':None,
                'train_name':None,
                'test_text':None,
                'test_name':None}
    
    word_id=data['word_id']
    train_text=[]
    for x in data['train_text']:
        digit=[]
        for y in x:
            if word_id.has_key(y):
                digit.append(word_id[y])
            else:
                digit.append(None)
        train_text.append(digit)
    digit_data['train_text']=train_text
    
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
    digit_data['train_name']=name_one_hot;
    
    test_text=[]
    for x in data['test_text']:
        digit=[]
        for y in x:
            if word_id.has_key(y):
                digit.append(word_id[y])
            else:
                digit.append(None)
        test_text.append(digit)
    digit_data['test_text']=test_text
    
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
    digit_data['test_name']=name_one_hot1
    return digit_data
    
import tensorflow as tf
class DataSet(object):
    def __init__(self,text,name,dtype=tf.float32):
        assert len(text) == name.shape[0], (
            'text.shape: %s name.shape: %s' % (len(text),
                                               name.shape))
        self._num_examples = len(text)
        self._text = text
        self._name = name
        self._text_len = [len(x) for x in text]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def text(self):
        x=np.zeros([self._num_examples,word_per_br,word_size],dtype=np.float32)
        i=0
        for a in self._text:
            j=0
            for b in a:
                if b is not None:
                    x[i,j,b]=1
                j=j+1
            i=i+1
        return x
        
    @property
    def name(self):
        return self._name
        
    @property
    def text_len(self):
        return np.array(self._text_len)
    
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
            self._text = [self._text[index] for index in perm]
            self._name = self._name[perm]
            self._text_len = [self._text_len[index] for index in perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        x=np.zeros([len(self._text[start:end]),word_per_br,word_size],dtype=np.float32)
        i=0
        for a in self._text[start:end]:
            j=0
            for b in a:
                if b is not None:
                    x[i,j,b]=1
                j=j+1
            i=i+1
        return x, self._name[start:end], np.array(self._text_len[start:end])
    
def read_data_sets(train_file_name, test_file_name,
                   dtype=tf.float32):
    class DataSets(object):
        pass
    data_sets = DataSets()
    
    data=read_data(train_file_name,test_file_name)
    digit_data=data_digital(data)
    
    data_sets.train=DataSet(digit_data['train_text'],
                            digit_data['train_name'])
    
    data_sets.test=DataSet(digit_data['test_text'],
                           digit_data['test_name'])
    return data_sets

'''
Vector_size=5000            #one_hot编码的长度
Word_size=100               #每个bug report中提取的词的长度

data_list=[]
word_num={}
name_id={}
###------------------------------------------------#
#           将文本文件以列表形式存到data_list
#           将词-频率对存入word_num
#           将维护人名字存入name_id
#------------------------------------------------###
train_file=open('eclipse_words_train.txt','r')
out_file=open('name_id.txt','w')
i=0
for line in train_file:
    line='line='+line
    exec line                           #line表示为list类型了
    data_list.append(line)
    if not name_id.has_key(line[2]):
        name_id[line[2]]=i;
        i=i+1
    for word in line[5]:
        if word_num.has_key(word):
            word_num[word]=word_num[word]+1
        else:
            word_num[word]=1
    for sentence in line[6]:
        for word in sentence:
            if word_num.has_key(word):
                word_num[word]=word_num[word]+1
            else:
                word_num[word]=1    
train_file.close()

#对词排序并将前5000个词用one_hot编码表示存入word_code
sorted_word_num=sorted(word_num.iteritems(),key=lambda t:t[1],reverse=True)
word_code={}
allzeros=np.zeros(Vector_size,dtype='float32')
i=0
for x in sorted_word_num:
    if i<5000:
        one_hot=np.zeros(Vector_size,dtype='float32')
        one_hot.flat[i]=1.
        word_code[x[0]]=one_hot
        i=i+1
        
#txt_trian是保存词序信息的bug report文本的one_hot编码版
txt_trian=np.zeros([len(data_list),Word_size,Vector_size],dtype='float32');
#name_train是保存维护者姓名的one_hot编码
name_train=np.zeros([len(data_list),len(name_id.keys())],dtype='float32');

i=0
for data in data_list:
    j=0
    if name_id.has_key(data[2]):
        name_train[i,name_id[data[2]]]=1.
    for word in data[5]:
        if j<100:
            if word_code.has_key(word):
                txt_trian[i,j]=word_code[word]
            j=j+1
    for sentence in data[6]:
        for word in sentence:
            if j<100:
                if word_code.has_key(word):
                    txt_trian[i,j]=word_code[word]
                j=j+1
    i=i+1
for x in name_id.items():
    out_file.writelines(str(x)+'\r\n')
out_file.close()
'''