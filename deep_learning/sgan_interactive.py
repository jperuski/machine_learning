# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:02:06 2018

@author: jperuski
"""

# baseline imports for python 3.6 using conda install

import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers import Activation
from keras.optimizers import sgd
from keras import losses
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as skld
from sklearn.datasets import load_wine

def define_d_model(n_feat):
    
    
        d_model = Model()
        x = Dense
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

        model.add(ZeroPadding2D(padding=((0,1),(0,1))))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))


        img = Input(shape=self.img_shape)



        features = model(img)

        valid = Dense(1, activation="sigmoid")(features)

        label = Dense(self.num_classes+1, activation="softmax")(features)



        return Model(img, [valid, label])
    
    d_model = Model()
    g_model = Model()
    model = Model()
    
    return model

def train_sgan(model, n_epoch, n_batch):
    
    for e in np.arange(n_epoch):
        for b in np.arange(n_batch):
            
    
    return loss_vec

############################## Load Data ####################################
    
X_raw, y_raw = load_wine(return_X_y = True)
