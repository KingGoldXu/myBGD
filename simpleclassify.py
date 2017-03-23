#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import read_tfidfvec
import tensorflow as tf

min_frequency=2
#restore=True

bugrpt=read_tfidfvec.read_data_sets('train0.txt','test0.txt',min_frequency=min_frequency)
input_num=bugrpt.train.textvec.shape[1]
output_num=bugrpt.train.namevec.shape[1]
x=tf.placeholder("float",[None,input_num])
W=tf.Variable(tf.random_uniform([input_num,output_num],-1.0,1.0), name="W")
b=tf.Variable(tf.constant(0.1,shape=[output_num]),name="b")
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder("float",[None,output_num])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
correct_pre = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pre,'float'))

#saver=tf.train.Saver()

#if restore:
    #saver.restore(sess,'./simpleclassify.ckpt')
    
for i in range(3000):
    batch_xs, batch_ys = bugrpt.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if i%100==0:
        print sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        #print sess.run(accuracy, feed_dict={x: bugrpt.train.textvec, y_: bugrpt.train.namevec})

#saver.save(sess,'simpleclassify.ckpt')
print sess.run(accuracy, feed_dict={x: bugrpt.test.textvec, y_: bugrpt.test.namevec})
