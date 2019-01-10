#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential, Model
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Lambda, GlobalAveragePooling2D
from keras.layers import Cropping2D,BatchNormalization, Conv2DTranspose, Dense, Flatten, GlobalMaxPooling2D, concatenate

from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import *
from keras.utils.vis_utils import plot_model



class MTbuilder(object):
    
    @staticmethod
      
    def model():
        img_input = Input(shape=(None, None, 1))
#        img_input = Input(shape=(192, 256, 1))

        conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(img_input)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        drop4 = Dropout(0.5)(conv4)
#        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#        drop5 = Dropout(0.5)(conv5)

#        up6 = Conv2D(256, 2 ,activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
#        merge6 = merge([drop4,up6], mode='concat', concat_axis=3)

        conv6 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(drop4)

        up7 = Conv2D(128, 2 ,activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3,up7], axis=-1)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

        up8 = Conv2D(64, 2 ,activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2,up8], axis=-1)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

        up9 = Conv2D(32, 2 ,activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1,up9], axis=-1)
        pool9 = MaxPooling2D(pool_size=(2, 2))(merge9)
        conv9 = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool9)
        conv10 = Conv2D(1, 1, padding='same', name='out1')(conv9)
        
        b1 = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        b2 = MaxPooling2D((2, 2))(b1)
        b3 = Conv2D(1024, 1, activation='relu', padding='same', kernel_initializer='he_normal')(b2)
        b4 = GlobalAveragePooling2D()(b3)
        b5 = Dense(256)(b4)
        clasification = Dense(units=5,
                          kernel_initializer="he_normal",
                          activation="softmax", name='out2')(b5)
        model = Model(inputs=img_input, outputs=[conv10, clasification], name='mtnet')
        model.summary()
        return model

    @staticmethod
    def build_mtnet():
        return MTbuilder.model()
    
       
if __name__ == '__main__':
    model = MTbuilder.build_mtnet()
    imgname = 'mtet.png'
    plot_model(model, to_file=imgname, show_shapes=True)
    

