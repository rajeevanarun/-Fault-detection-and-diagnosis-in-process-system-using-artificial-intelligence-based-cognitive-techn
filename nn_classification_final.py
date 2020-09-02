# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:49:31 2018

@author: rajee
"""
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

model_path = 'E:\\MUN PhD work\\My_Review\\9. 29th November\\NN_Classification\\NN_model.ckpt'
# input
#[x_train,x_test,y_train,y_test] = get_data()
# output one hot

#y_test = pd.get_dummies(y_test, sparse=True)

# Neural network model


#batch_size = 100
#samples = x_train.shape[0]

# place hold will usefull for specific datatype and shape


def train_NN(x_train, y_train,label):
    
# Neural Network model
    n_input = x_train.shape[1]
    hidden_layer = 60
    n_classes = label
    y_train = pd.get_dummies(y_train, sparse=True)
#
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None,n_input]) # datatype, shape 
    y = tf.placeholder(tf.float32,[None,n_classes])
    
    weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, hidden_layer])),
    'out': tf.Variable(tf.truncated_normal([hidden_layer,n_classes])) 
    }
    biases = {
    'b1': tf.Variable(tf.random_normal([hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # hidden layer
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
#output layer
    y_hat = tf.add(tf.matmul(layer_1,weights['out']), biases['out'])
    
    cost_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = y))
#optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost_fun) 
    optimizer = tf.train.AdamOptimizer(0.08).minimize(cost_fun)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    sess = tf.Session()
    sess.run(init)
    
#    mse_history = []
    accuracy_history =[]
    cost_history = []
#    print(sess.run(weights))
#    print(sess.run(biases))    
    
    for epoch in range(60):
        sess.run(optimizer, feed_dict = {x: x_train, y:y_train})
        cost = sess.run(cost_fun, feed_dict = {x: x_train, y:y_train})
        cost_history = np.append(cost_history,cost)
        correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        accuracy = (sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
        accuracy_history.append(accuracy)
        
#        pred_y = sess.run(y_hat,feed_dict = {x:x_test})
#        mse = tf.reduce_mean(tf.square(pred_y - y_test))
#        mse_ = sess.run(mse)
#        mse_history.append(mse_)
#    print(sess.run(weights))
#    print(sess.run(biases))
    
    save_path = saver.save(sess, "neuralnet/NL.ckpt")
    print("Model saved in path: %s" % save_path)
    print('epoch: ', epoch, '-', 'cost', cost, "- Train Accuracy: ", accuracy)
    
    plt.plot(accuracy_history)
    plt.show()
#    
    return

def test_NN(x_test,label):
    n_input = x_test.shape[1]
    hidden_layer = 60
    n_classes = label
    
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, [None,n_input]) # datatype, shape 
#    y = tf.placeholder(tf.float32,[None,n_classes])
    
    weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, hidden_layer])),
    'out': tf.Variable(tf.truncated_normal([hidden_layer,n_classes])) 
    }
    
    biases = {
    'b1': tf.Variable(tf.random_normal([hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
#output layer
    y_hat = tf.add(tf.matmul(layer_1,weights['out']), biases['out'])

    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()
    with tf.Session ()as sess:
        sess.run(init)
        saver.restore(sess, "neuralnet/NL.ckpt")
        y_hat = sess.run(y_hat,feed_dict = {x:x_test})
#        print (sess.run(weights))
#        print (sess.run(biases))
        prediction = tf.argmax(y_hat,1)
        
#        correct_prediction = tf.equal(prediction, tf.argmax(y_test,1))
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#    
        prediction_run = sess.run(prediction,feed_dict={x:x_test})
#        accuracy_run = sess.run(accuracy,feed_dict={x:x_test,y: y_test})
        print(prediction_run)
#        y_test_d = sess.run (tf.argmax(y_test, axis=1))
    
#    cnf_matrix = confusion_matrix(y_test_d, prediction_run)
#    
#    print(cnf_matrix)
#    
#    skplt.metrics.plot_confusion_matrix(y_test_d, prediction_run, normalize=True)
#
#    plt.show()
#    print("Train Accuracy: ", accuracy_run)
    return prediction