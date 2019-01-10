#from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout
#from keras.layers import Cropping2D,BatchNormalization, Conv2DTranspose, Dense, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import os 
import cv2
import keras.backend as K
import math
from MTnet import MTbuilder


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

dataset = 'B'
train_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train/'
train_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train_den/'
val_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val/'
val_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val_den/'
img_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
den_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/ground_truth_csv/'

def mae1(y_true, y_pred):
    return abs(K.sum(y_true) - K.sum(y_pred))
def mse1(y_true, y_pred):
    return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))

def data_pre_train():
    print('loading data from dataset ', dataset, '...')
    train_img_names = os.listdir(train_path)
    img_num = len(train_img_names)

    train_data = []
    for i in range(img_num):
        if i % 500 == 0:
            print(i, '/', img_num)
        name = train_img_names[i]
        img = cv2.imread(train_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        den = np.loadtxt(open(train_den_path + name[:-4] + '.csv'), delimiter = ",")
        den_quarter = np.zeros((int(den.shape[0] / 2), int(den.shape[1] / 2)))
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(2):
                    for q in range(2):
                        den_quarter[i][j] += den[i * 2 + p][j * 2 + q]

        count = np.sum(den)
        if count < 1:            
            den_class = 0
        elif count < 5:
            den_class = 1
        elif count < 10:
            den_class = 2
        elif count < 20:
            den_class = 3
        else:
            den_class = 4
        train_data.append([img, den_quarter, den_class])

    print('load train data finished.')
    return train_data

def data_pre_test():
    print('loading test data from dataset', dataset, '...')
    img_names = os.listdir(img_path)
    img_num = len(img_names)

    data = []
    for i in range(img_num):
        if i % 50 == 0:
            print(i, '/', img_num)
        name = 'IMG_' + str(i + 1) + '.jpg'

        img = cv2.imread(img_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128

        den = np.loadtxt(open(den_path + name[:-4] + '.csv'), delimiter = ",")
        den_quarter = np.zeros((int(den.shape[0] / 2), int(den.shape[1] / 2)))
        #print(den_quarter.shape)
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(2):
                    for q in range(2):
                        den_quarter[i][j] += den[i * 2 + p][j * 2 + q]
                        
        
        count = np.sum(den)
        if count < 50:            
            den_class = 0
        elif count < 100:
            den_class = 1
        elif count < 200:
            den_class = 2
        elif count < 400:
            den_class = 3
        else:
            den_class = 4
        data.append([img, den_quarter, den_class])
   
    print('load test data finished.')
    return data

data = data_pre_train()
data_test = data_pre_test()
np.random.shuffle(data)

x_train = []
y_train = []
y_train_class = []
for d in data:
    x_train.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_train.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))
    y_train_class.append(d[2])
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train_class = np.array(y_train_class)
y_train_class = to_categorical(y_train_class)

x_test = []
y_test = []
y_test_class = []
for d in data_test:
    x_test.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_test.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))
    y_test_class.append(d[2])
x_test = np.array(x_test)
y_test = np.array(y_test)

y_test_class = np.array(y_test_class)
y_test_class = to_categorical(y_test_class)


model = MTbuilder.build_mtnet()
adam = Adam(lr = 1e-4)
model.compile(optimizer= adam,
              loss={'out1': 'mse', 'out2': 'categorical_crossentropy'},
              loss_weights={'out1': 1., 'out2': 0.00001}, metrics = {'out1': [mae1, mse1], 'out2': 'accuracy'})

best_mae = 10000
best_mae_mse = 10000
best_mse = 10000
best_mse_mae = 10000
for i in range(200):
    model.fit(x_train, [y_train, y_train_class], epochs = 3, batch_size = 1, validation_split = 0.2)
    score = model.evaluate(x_test, [y_test, y_test_class], batch_size = 1)
    score[4] = math.sqrt(score[4])
    print(score)
    if score[3] < best_mae:
        best_mae = score[3]
        best_mae_mse = score[4]
        
        json_string = model.to_json()
        open('model_mt.json', 'w').write(json_string)
        model.save_weights('weights_mt.h5')
    if score[4] < best_mse:
        best_mse = score[4]
        best_mse_mae = score[3]

    print('best mae: ', best_mae, '(', best_mae_mse, ')')
    print('best mse: ', '(', best_mse_mae, ')', best_mse)
    







