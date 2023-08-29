import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import seaborn as sns
import warnings
idx= pd.IndexSlice
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')


keys = ['TSLA' , 'GOOG' , 'META' , 'AAPL' , 'MSFT' , 'AMZN' , 'NVDA' , 'V' , 'ORCL' , 'CSCO']


# def model1(in_len,out_len):
#     main_input = tf.keras.layers.Input(shape=(in_len,))
#     m = (Dense(512, activation='tanh')) (main_input)
#     m = (Dropout(0.2))(m)
#     m =(Dense(256, activation='tanh'))(m)
#     m = (Dropout(0.2))(m)
#     m =(Dense(128, activation='tanh')) (m)
#     m = (Dropout(0.2))(m)
#     m =(Dense(64, activation='tanh')) (m)
#     cnn_input = tf.keras.layers.Reshape((in_len, 1))(main_input)
#     cnn = tf.keras.layers.Conv1D(128, 2, padding='same', strides=1, activation='relu')(cnn_input)
#     cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)

#     flat = tf.keras.layers.Flatten()(cnn)
#     flat = tf.keras.layers.Concatenate(axis=-1)([flat , m])
#     drop = tf.keras.layers.Dropout(0.2)(flat)
    
#     dense1 = tf.keras.layers.Dense(512, activation='relu')(drop)
#     main_output = tf.keras.layers.Dense(out_len, activation='softmax')(dense1)
#     model = tf.keras.Model(inputs=main_input, outputs=main_output)
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     return model
# model1 = get_model1(X.shape[1])



def model2(in_len,out_len):
    x = Input(( in_len ,))
    m = (Dense(512, activation='tanh')) (x)
    m = (Dropout(0.2))(m)
    m =(Dense(256, activation='tanh'))(m)
    m = (Dropout(0.2))(m)
    m =(Dense(128, activation='tanh')) (m)
    m = (Dropout(0.2))(m)
    m =(Dense(64, activation='tanh')) (m)
    m =(Dense(128, activation='tanh'))(m)
    m =(Dense(256, activation='tanh'))(m)
    m = (Dense(512, activation='tanh'))(m)

    t = tf.keras.layers.Concatenate(axis=-1)([x , m])
    t =(Dense(256, activation='tanh'))(t)
    f = (Dense(out_len, activation='softmax')) (t)

    model = Model(inputs=x, outputs=f)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

#model2 = get_model2()




    
