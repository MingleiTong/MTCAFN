#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:51:35 2018

@author: tongml
"""
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import sys
import os 
import cv2
import math
import scipy.io as io

dataset = 'B'
img_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
den_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/ground_truth_csv/'

def data_pre_test():
    print('loading test data from dataset', dataset, '...')
    img_names = os.listdir(img_path)
    img_names = img_names
    img_num = len(img_names)

    data = []
    for i in range(1, img_num + 1):
        if i % 50 == 0:
            print(i, '/', img_num)
        name = 'IMG_' + str(i) + '.jpg'
        img = cv2.imread(img_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        den = np.loadtxt(open(den_path + name[:-4] + '.csv'), delimiter = ",")
        den_sum = np.sum(den)
        data.append([img, den_sum])
            
    print('load data finished.')
    return data
    
data = data_pre_test()

model = model_from_json(open('./model_mt.json').read())
model.load_weights('./weights_mt.h5')

mae = 0
mse = 0
ii = 0
matfile = []
for j in range(len(data)):
    ii += 1
    inputs = np.reshape(data[j][0], [1, data[j][0].shape[0], data[j][0].shape[1], 1])
    den = data[j][1]

    outputs = model.predict(inputs)   

    den = data[j][1]
    c_act = den
    c_pre = np.sum(outputs)
    t1 = int(c_act)
    t2 = int(c_pre)
    matfile.append([t1, t2])
    print('no', ii , 'pre:', c_pre, 'act:', c_act)
    mae += abs(c_pre - c_act)
    mse += (c_pre - c_act) * (c_pre - c_act)
mae /= len(data)
mse /= len(data)
mse = math.sqrt(mse)

print('#############################')
print('mae:', mae, 'mse:', mse)