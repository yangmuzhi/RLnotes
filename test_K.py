#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:57:17 2019

@author: muzhi
"""

import keras.backend as K

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam

def model():
    inputs = Input(shape=(4,))
    x = Dense(32)(inputs)
    x = Dense(1, activation="softmax")(x)
    return Model(inputs,x)

m = model()
m.summary()
m.output
x = K.placeholder((None,4))
y = K.placeholder((None,4))
K.concatenate([x,m.input])
K.concatenate([x,y])












