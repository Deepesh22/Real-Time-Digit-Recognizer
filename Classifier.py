#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 20:02:07 2018

@author: deepesh

-> Algorithms: 
    1. Softmax Regression - Multi-class Classification [0 to 9 i.e 10 classes]
    2. Gradient Descent - Optimization
    3. Back-Propagation - Adjustment of the weights by calculating the gradient of the loss function
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

def tderivative(z):
    return 1 - tanh(z)**2

def softmax(z):
    z -= np.max(z)
    softmax = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return softmax

def Model(x_train, y_train, y_train_true, x_test, y_test, y_test_true,learning_rate = 1, iteration= 10000, print_cost = False, show_plot = False):
    e = []
    m = x_train.shape[1]
    np.random.seed()
    global weight
    global bias
    #weight bias initialization
    #
    #input layer -> 784 units
    #hidden layer 1 -> 16 units with activation function tanh
    #ouput layer -> 16 units with activation function softmax
    #
    w1 = np.random.randn(16,784) * np.sqrt(1/784) #weight matrix to hidden layer 1
    b1 =  np.random.randn(16, 1) * np.sqrt(1/784)
    w2 =  np.random.randn(10, 16) * np.sqrt(1/16) #weight matrix to output layer 
    b2 = np.random.randn(10, 1)* np.sqrt(1/16)

    for index in range(1,iteration+1):
        """ Forward propagation"""
        
        Z1 = np.dot(w1,x_train) + b1
        A1 = tanh(Z1)
        
        Z2 = np.dot(w2,A1) + b2
        A2 = softmax(Z2.T)

        "COST (cross-entropy)"
        logprobs = y_train*np.log(A2) + (1-y_train)*np.log(1-A2)
        cost = -np.sum(logprobs)/m
        cost = np.squeeze(cost) 

        """ Backward Propagation"""
        dZ2 = A2 - y_train
        dW2 = (1/m)*np.dot(dZ2,np.transpose(A1))
        db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims=True)
        
        dZ1 = np.dot(np.transpose(w2), dZ2) * tderivative(Z1)
        dW1 = (1/m)*np.dot(dZ1,np.transpose(x_train))
        db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims=True)
        
        """ Weight bias update"""
        w1 = w1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        w2 = w2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        

        weight = {"w1":w1, "w2":w2}
        bias = {"b1":b1, "b2":b2}  

        if print_cost and index%100 == 0:
            print("Iterations complete ", index, ": Error ", np.sum(cost))

            #train_pred = predict(weight, bias, x_train).reshape(33600,)
            #test_pred = predict(weight, bias, x_test).reshape(8400,)
            #print("Training Phase accuracy percentage: ", (100*list(train_pred-y_train_true.reshape(33600,)).count(0))/(float(len(train_pred))))
            #print("Testing Phase accuracy percentage: ", list(test_pred-y_test_true.reshape(8400,)).count(0)/(float(len(test_pred)))*100)
        e.append(np.sum(cost))        

    if show_plot: # plot for cost vs number of iterations
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost vs Iterations")
        plt.plot(range(iteration), e)
        plt.show()

    train_pred = predict(weight, bias, x_train).reshape(33600,)
    test_pred = predict(weight, bias, x_test).reshape(8400,)

    print("Training Phase accuracy percentage: ", list(train_pred-y_train_true.reshape(33600,)).count(0)/336)
    print("Testing Phase accuracy percentage: ", list(test_pred-y_test_true.reshape(8400,)).count(0)/84)

    return weight, bias

def predict(weight, bias, x):
    w1 = weight["w1"]
    w2 = weight["w2"]
    b1 = bias["b1"]
    b2 = bias["b2"]

    l1 = tanh(np.dot(w1,x) + b1)
    l = softmax((np.dot(w2,l1) + b2).T)
    y_pred = np.argmax(l, axis=0)
    return y_pred
