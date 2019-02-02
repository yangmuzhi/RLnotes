from utils.net import conv_breakout
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Flatten
import tensorflow as tf
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def data_gen():

    while True:
        x = np.random.random((32,28,28,3))
        y = np.random.uniform(0,1,32)
        yield x, y

def conv():
    inputs = Input((28,28,3))
    x = conv_breakout(inputs)
    x = Dense(1)(x)
    return Model(inputs, x)
data = data_gen()


model = conv()
model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mse'])
model.fit_generator(data,epochs=100, steps_per_epoch=100)
