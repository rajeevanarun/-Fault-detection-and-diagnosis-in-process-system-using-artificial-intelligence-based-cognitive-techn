# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 20:07:22 2018

@author: rajee
"""
import numpy as np
import matplotlib.pyplot as plt

from de import get_data
from dic_onc import tf_OneClass_NN_sigmoid_Train
from dic_onc import tf_OneClass_NN_sigmoid_Test
from nn_classification_final import train_NN
from nn_classification_final import test_NN
import pandas as pd

[x_train,x_test] = get_data()
label = 1
ap_arr = []
ap_arr_train = []
ap_arr_test = []
ap_arr_test1 = []
window_size = 100
anomaly_points1 = 0
test_result1 = []
n = 0

def train_model(x_train,y_train,label):
    print(len(x_train))
    print(len(y_train))
#train_NN(x_train, y_train)
    train_result = tf_OneClass_NN_sigmoid_Train(x_train)
    train_result = train_result[0]
    le = len(train_result)
    for i1 in range(1,le,window_size):
        anomaly_points_train = sum(n1<0 for n1 in train_result[i1:i1+window_size])
        ap_arr_train.append(anomaly_points_train)
        
    m = np.max(np.asarray(ap_arr_train, dtype = int))
    margin = m + (0.5 * m)
    #print("margin :", margin)
    train_NN(x_train, y_train,label)
    return (margin,x_train)

# Test Model
# Check the anomalities and train the NN
def test_model(x_test,x_train,margin,y_train):
    global label
    length = len(x_test)
    print(len(x_test))
    for i in range(0,length,window_size):
        print(i)
        test_result = tf_OneClass_NN_sigmoid_Test(x_test[i:i+window_size,:])
        ap = sum(n1 < 0 for n1 in test_result)
        anomaly_points = sum(ap)
        ap_arr_test.append(anomaly_points)
        
        if (anomaly_points > margin) : 
            for check in range (i+window_size,i+2*window_size,window_size):
                test_result1 = tf_OneClass_NN_sigmoid_Test(x_test[check:check+window_size,:])
                ap1 = sum(nn < 0 for nn in test_result1)
                anomaly_points1 = sum(ap1)
                ap_arr_test1.append(anomaly_points1)
                
                if (anomaly_points1 > margin):
                    print(margin)
                    label = label + 1
                    print("Fault", label) 
                    print(i)
                    print(check)
                    y_t = label * np.ones(len(x_test[i: check+window_size]))
                    y_train = np.concatenate((y_train,y_t),axis=0)
                    x_train = np.concatenate((x_train, x_test[i: check+window_size]), axis=0)
                    margin,x_train = train_model(x_train, y_train,label)  
                else:
                    print("false")
                    test_NN(x_test[i: check+window_size],label)
        else:
            print("false")
            test_NN(x_test[i:i+window_size,:],label)
    return
def main(x_train):
    label = 1
    y_train = np.ones(len(x_train))
    margin,x_train = train_model(x_train, y_train,label)
    test_model (x_test,x_train,margin,y_train)
    return