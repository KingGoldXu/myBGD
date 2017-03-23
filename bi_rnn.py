#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import input

bugrpt=input.read_data_sets('train0.txt','test0.txt')

restore=True

learn_rate=0.002
batch_size=100

num_input=5000
num_step=100
gru_size=500
num_classes=bugrpt.train.name.shape[1]

inputs=tf.placeholder("float",[None,num_step,num_input])
y=tf.placeholder("float",[None,num_classes])
input_lens=tf.placeholder("int32",[None,])
keep_prob=tf.placeholder(tf.float32)

W=tf.Variable(tf.random_uniform([2*gru_size,num_classes],-1.0,1.0), name="W")
b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")

inputs_drop=tf.nn.dropout(inputs, keep_prob)
#BI_RNN
d_gru_cell=tf.contrib.rnn.GRUCell(gru_size)
d_gru_cell=tf.contrib.rnn.DropoutWrapper(d_gru_cell,
                                         output_keep_prob=keep_prob)
    
gru_outputs,_=tf.nn.bidirectional_dynamic_rnn(
    cell_fw=d_gru_cell,
    cell_bw=d_gru_cell,
    dtype=tf.float32,
    sequence_length=input_lens,
    inputs=inputs_drop)
    
(gru_fw_outputs,gru_bw_outputs)=gru_outputs
    
h_fw = tf.nn.max_pool(tf.reshape(gru_fw_outputs,[-1,num_step,gru_size,1]),
                      [1,num_step,1,1],[1,num_step,1,1],padding='VALID')
    
h_bw = tf.nn.max_pool(tf.reshape(gru_bw_outputs,[-1,num_step,gru_size,1]),
                      [1,num_step,1,1],[1,num_step,1,1],padding='VALID')

h = tf.concat([h_fw, h_bw],2)
h = tf.reshape(h,[-1,2*gru_size])

h = tf.nn.dropout(h,keep_prob)

pred = tf.matmul(h,W)+b
#定义cost和optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
#计算准确度评估模型
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#初始化变量
init = tf.global_variables_initializer()
#开始训练模型
with tf.Session() as sess:
    sess.run(init)
    saver=tf.train.Saver()

    if restore:
        saver.restore(sess,'./log/bi_rnn.ckpt')
    for i in range(50):
        batch_xs, batch_ys ,batch_inputlens = bugrpt.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={inputs: batch_xs,
                                       y: batch_ys,
                                       input_lens: batch_inputlens,
                                       keep_prob: 0.6})
        
        if i%10==9:
            print(sess.run(accuracy, feed_dict={inputs: batch_xs,
                                                y: batch_ys,
                                                input_lens: batch_inputlens,
                                                keep_prob: 0.6}))
    saver.save(sess,'log/bi_rnn.ckpt')        
    print(sess.run(accuracy, feed_dict={inputs: bugrpt.test.text,
                                        y: bugrpt.test.name ,
                                        input_lens: bugrpt.test.text_len,
                                        keep_prob: 1.0}))