 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: wuzhenglin
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import scipy.stats

# make the label of dataset
def change(arr):
    
    
    inti = arr[0]
    
    if inti == 0:
        inti = np.array([1, 0])
    else:
        inti = np.array([0, 1])
    
    inti = inti[np.newaxis, :] 
    
    for i in range(1, arr.shape[0]):
        
        tem = arr[i]
    
        if tem == 0:
            tem = np.array([1, 0])
        else:
            tem = np.array([0, 1])
        
        tem = tem[np.newaxis, :] 
    
        inti = np.append(inti, tem, axis = 0)
        
    return inti
        
        
        
# read the dataset from the tfrecords
def read_and_decode(p):
    


    filename_queue = tf.train.string_input_producer([p]) 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   

    features = tf.parse_single_example(serialized_example,
                                       features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                               })  

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [128, 128])
    label = tf.cast(features['label'], tf.int32)
    
    
    
    return image, label

# Begin the CNN building 
def weight_variable(path, select, shape):
    
    if select == 1:
        initial = np.fromfile(path, dtype = np.float32)
        initial = initial.reshape(shape)
        

    else:        
        initial = tf.truncated_normal(shape, stddev = 0.1, dtype = tf.float32)
        
    return tf.Variable(initial)


#find index in delta_weight
def find_max(a):
    a = abs(a)
    index_a = a.argmax(axis = 0)
    
    if a[index_a[0]][0] > a[index_a[1]][1]:
        return index_a[0]
    else:
        return index_a[1]
    
def bias_variable(path, select, shape):
    
    if select == 1:
        initial = np.fromfile(path, dtype = np.float32)
        initial = initial.reshape(shape)
    
    else:        
        initial = tf.constant(0.1, shape = shape, dtype = tf.float32)

    return tf.Variable(initial)


def conv2d(x, W):
    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  



def cnn_model():

    hidden = 256
    image_size = 128 
    label_size = 2
    
    hc_epochs = 3
    hc_batch_size = 26
    hc_learning_rate = 0.0005       
    cap_c = 1214
    cap_h = 1290
    hc_num = 200
    
    c_epochs = 10   
    c_batch_size = 15
    c_learning_rate = 0.00008
    c_num = 100
    
    hc_step = ((hc_num//(hc_batch_size)) * hc_epochs)
    c_step = ((c_num//(c_batch_size)) * c_epochs)
    
    print '@healthyORcancer  The number of training steps: ', hc_step
    print '@Contrast  The number of training steps: ', c_step
    
    #data set
    hc_train = '/media/stargazer/Hurt Cloud/dataset/body/tfrecords/braindata/brain.tfrecords'
    hc_im, hc_lab = read_and_decode(hc_train)
    hc_train_img, hc_train_label = tf.train.shuffle_batch([hc_im, hc_lab],
                                            batch_size = hc_batch_size, capacity = hc_num,
                                            min_after_dequeue = 10)

    hc_test = '/media/stargazer/Hurt Cloud/dataset/body/tfrecords/braindata/brain_test.tfrecords'
    hc_im_, hc_lab_ = read_and_decode(hc_test)  
    hc_test_img, hc_test_label = tf.train.shuffle_batch([hc_im_, hc_lab_],
                                            batch_size = 15, capacity = 20,
                                            min_after_dequeue = 10)
    
    c_train = '/media/stargazer/Hurt Cloud/dataset/body/tfrecords/braindata/brain_gender.tfrecords'
    c_im, c_lab = read_and_decode(c_train)
    c_train_img, c_train_label = tf.train.shuffle_batch([c_im, c_lab],
                                                batch_size = c_batch_size, capacity = c_num,
                                                min_after_dequeue = 10)
    
    c_test = '/media/stargazer/Hurt Cloud/dataset/body/tfrecords/braindata/brain_gender_test.tfrecords'
    c_im_, c_lab_ = read_and_decode(c_test)  
    c_test_img, c_test_label = tf.train.shuffle_batch([c_im_, c_lab_],
                                            batch_size = 15, capacity = 20,
                                            min_after_dequeue = 5)


    #matrix for plot acc and loss
    hc_train_loss = np.empty(hc_step)
    hc_train_acc = np.empty(hc_step)
    
    c_train_loss = np.empty(c_step)
    c_train_acc = np.empty(c_step)
    
    ind_dis = np.empty(c_step)
    
    #matrix for plot weight of last layer
    f, hch_plots = plt.subplots(int((hc_step + c_step) / 4) + 1, 4, figsize=(100, 100))
    
    #matrix for plot loss and accuracy
    ff, loss_acc_plots = plt.subplots(2, 2)
   
    hc = 0   
    c = 1
    chc_select = 0 
    
    #0: healthy and cancer of lung
    #1: contrast of lung
    #2: do the test after modify W4 weight matrix
    for chc in range(3):
        
        
        if chc_select == hc or chc == 2:
        
                print 'Healthy and Cancer Section'
                epochs = hc_epochs
                batch_size = hc_batch_size
                learning_rate = hc_learning_rate
                num = hc_num
                stepnum = hc_step
                
                train_image = hc_train_img
                train_label = hc_train_label
                test_image = hc_test_img
                test_label = hc_test_label
                num_test = 15
                
                train_loss = hc_train_loss
                train_acc = hc_train_acc
        
        if chc_select == c:
                
                print 'Contrast Section'
                epochs = c_epochs
                batch_size = c_batch_size
                learning_rate = c_learning_rate
                num = c_num
                stepnum = c_step
                
                train_image = c_train_img
                train_label = c_train_label
                test_image = c_test_img
                test_label = c_test_label
                num_test = 15
                
                train_loss = c_train_loss
                train_acc = c_train_acc        
            
    
        #Model Parameter
        x = tf.placeholder(tf.float32, shape = [None, image_size * image_size])
        y = tf.placeholder(tf.float32, shape = [None, label_size])
    
        weight_balance = tf.constant([0.1])
  
        X_train_ = tf.reshape(x, [-1, image_size, image_size, 1])

        #First layer
        W_conv1 = weight_variable("w1.bin", chc_select, [5, 5, 1, 6])
        b_conv1 = bias_variable("b1.bin", chc_select, [6])
      
        h_conv1 = tf.nn.relu(conv2d(X_train_, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
    
        #Second layer
        W_conv2 = weight_variable("w2.bin", chc_select, [5, 5, 6, 16])
        b_conv2 = bias_variable("b2.bin", chc_select, [16])
        
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Third layer
        W_conv5 = weight_variable("w5.bin", chc_select, [5, 5, 16, 120])
        b_conv5 = bias_variable("b5.bin", chc_select, [120])

        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

        
        #Full connect layer
        W_fc1 = weight_variable("w3.bin", chc_select, [32 * 32 * 120, hidden])
        b_fc1 = bias_variable("b3.bin", chc_select, [hidden])

        h_pool5_flat = tf.reshape(h_conv5, [-1, 32 * 32 * 120])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

        #drop layer
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #Output_Softmax
        W_fc2 = weight_variable("w4.bin", chc_select, [hidden, label_size])
        b_fc2 = bias_variable("b4.bin", chc_select, [label_size])

        out_feed = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)
        y_conv = tf.nn.softmax(out_feed)


        #Train
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, out_feed, weight_balance))
        optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        #Sess section
        print 'Begin training'
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init_op)

            print 'Build the thread'
            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(coord = coord)
            
            

                        
            if chc != 2:
                #Begin the training process
                step = 1
                
                for ep in range(epochs):
                
                    for i in range(num//batch_size):
                        
                        example, l = sess.run([train_image, train_label])
                        example_, l_ = sess.run([test_image, test_label])
                        
                        ll = change(l)
                        ll_ = change(l_)
                        
                        example = example.flatten()
                        example = example.reshape([batch_size, image_size * image_size])
                        example_ = example_.flatten()
                        example_ = example_.reshape([num_test, image_size * image_size])
                    
                        feed_dict = {x: example, y: ll, keep_prob: 1.0}
                        feed_dict_ = {x: example_, y: ll_, keep_prob: 1.0}
                        feed_dict_d = {x: example, y: ll, keep_prob: 0.3}
                        
                        sess.run(optimize, feed_dict = feed_dict_d)
                        
                        if step == 1:
                            
                            weight_4 = np.array([[0, 0],
                                                 [0, 0],
                                                 [0, 0],
                                                 [0, 0]])


                        #before
                        weight_4_before = weight_4
                        
                        los, acc, weight_4 = sess.run([loss, accuracy, W_fc2], feed_dict = feed_dict)
                        p = sess.run([y_conv], feed_dict = feed_dict)
                        
                        if step == 1:
                            weight_4_before = weight_4
                        
                        #after
                        weight_4_delta = weight_4_before - weight_4
                        
                        if step != 1:
                            train_loss[step -2] = los
                            train_acc[step -2] = acc
        
                        if step == stepnum and chc_select == hc:
                            
                            w1, b1, w2, b2, w5, b5, w3, b3, w4, b4 = sess.run([W_conv1, b_conv1, W_conv2, b_conv2, W_conv5, b_conv5, W_fc1, b_fc1, W_fc2, b_fc2], feed_dict = feed_dict)
                            w1.tofile("w1.bin")
                            b1.tofile("b1.bin")
                            w2.tofile("w2.bin")
                            b2.tofile("b2.bin")
                            w5.tofile("w5.bin")
                            b5.tofile("b5.bin")
                            w3.tofile("w3.bin")
                            b3.tofile("b3.bin")
                            w4.tofile("w4.bin")
                            b4.tofile("b4.bin")
    
                        if chc_select == hc:
                            
                            print "@healthyORcancer  step: %d  loss: %.9f  accuracy: %.3f" % (step, los, acc)
                            hch_plots[int((step - 1) / 4), int(((step - 1) % 4))].axis('off')
                            hch_plots[int((step - 1) / 4), int(((step - 1) % 4))].imshow(weight_4_delta, cmap = plt.cm.YlGn)
                        
                        if chc_select == c:
                            
                            print "@Contrast  step: %d  loss: %.9f  accuracy: %.3f" % (step, los, acc)
                            hch_plots[int(int(((step + hc_step) - 1) / 4)), ((step + hc_step - 1) % 4)].axis('off')
                            hch_plots[int(int(((step + hc_step) - 1) / 4)), ((step + hc_step - 1) % 4)].imshow(weight_4_delta, cmap=plt.cm.Blues)
                            
                            ind = find_max(weight_4_delta)
                            print 'max', ind
                            ind_dis[step - 1] = ind
                            
                        
                        
#                        print 'weight_4_delta:\n', weight_4_delta
                        
#                        print 'pred:\n', p[0]
#                        print 'label:\n', ll
                        
                        if step == stepnum:
                                                            
                            acc_ = sess.run([accuracy], feed_dict = feed_dict_)
                            fucker = sess.run([y_conv], feed_dict = feed_dict)
                            
                            
                            print 'predicted result, before modifying:\n', fucker
                            print 'REAL result, before modifying:\n', ll_
                            print 'ACC：', acc_
                       
                        step = step + 1
                        
                                
    
         
                #Change the dataset to Contrast
        
            else:
                print 'Second Test：'
                 
                print 'mode', ind_dis
                print type(ind_dis)
                
                example, l = sess.run([train_image, train_label])
                example_, l_ = sess.run([test_image, test_label])
                
                ll = change(l)
                ll_ = change(l_)
                
                example = example.flatten()
                example = example.reshape([batch_size, image_size * image_size])
                example_ = example_.flatten()
                example_ = example_.reshape([num_test, image_size * image_size])
            
                feed_dict = {x: example, y: ll, keep_prob: 1.0}
                feed_dict_ = {x: example_, y: ll_, keep_prob: 1.0}
                feed_dict_d = {x: example, y: ll, keep_prob: 0.3}
            
                los, acc, weight_4 = sess.run([loss, accuracy, W_fc2], feed_dict = feed_dict)
                                
                select = scipy.stats.mode(ind_dis).mode[0]
                select = int(select)
                print 'remove it', select
                
                
                modify_w4 = sess.run([W_fc2], feed_dict = feed_dict)
                print '***************'
                print 'original w4', modify_w4[0]
                print '***************'
                modify_w4[0][select] = [0, 0]
                assign = W_fc2.assign(modify_w4[0])
                sess.run(assign)
                modify_w4_ = sess.run([W_fc2], feed_dict = feed_dict)
                acc_m = sess.run([accuracy], feed_dict = feed_dict_)
                fuckerm = sess.run([y_conv], feed_dict = feed_dict)
                print 'Modify w4', modify_w4_[0]
                print '***************'
                print 'predicted result, after modifying:\n', fuckerm
                print 'REAL result, after modifying:\n', ll_
                print 'ACC_：', acc_m
            
            chc_select = 1
            
            
            coord.request_stop()
            coord.join(threads)

    

if __name__ == '__main__':
    
    cnn_model()
