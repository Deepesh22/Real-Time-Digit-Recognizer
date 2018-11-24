#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:02:07 2018

@author: deepesh

for current parameters
Training Phase accuracy percentage:  95.26488095238095
Testing Phase accuracy percentage:  93.46428571428571 
"""

import cv2
import numpy as np
import pandas as pd
import Classifier

dataset = pd.read_csv("Data.csv")

x_train = dataset.iloc[:33600, 1:785].values/255
y_train = dataset.iloc[:33600, 0:1].values
y_train_true = y_train

x_test = dataset.iloc[33600:, 1:785].values/255
y_test = dataset.iloc[33600:, 0:1].values
y_test_true = y_test

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features=[0])
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.fit_transform(y_test).toarray()

# train model
weight, bias = Classifier.Model(x_train.T, y_train.T, y_train_true, x_test.T, y_test.T, y_test_true, 0.8, 2000, print_cost = True, show_plot = False)

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret , img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting to grayscale

    blur = cv2.GaussianBlur(gray, (35,35), 0) #blurring

    #thresholding
    ret, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    x, y, w, h = 0, 0, 300, 300
    thresh  = thresh[y:y + h, x:x + w]
    cv2.imshow("t", thresh)

    Captured_Image = cv2.resize(thresh, (28,28)) #resizing in 28 * 28 shape
    Captured_Image = np.array(Captured_Image)/255  # creating numpy array an dividing by 255 to normalise
    Captured_Image = Captured_Image.flatten() #flatten 
    Captured_Image = Captured_Image.reshape(Captured_Image.shape[0], 1) #reshape vector to shape (784,1)

    answer= Classifier.predict(weight, bias, Captured_Image) #predict result

    cv2.rectangle(img, (0,0), (300,300), (255,0,0))
    cv2.putText(img, "Predicted Digit is " + str(answer), (30, 320),cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Window",img)
    
    k=cv2.waitKey(30) & 0xFF #wait for esc to be pressed
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
