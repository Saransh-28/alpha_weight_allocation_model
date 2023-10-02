import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import warnings
idx= pd.IndexSlice
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
    
    #tanh used because it is better to be approximatly right than being precisely wrond
    t =(Dense(256, activation='tanh'))(t)
    f = (Dense(out_len, activation='softmax')) (t)

    model = Model(inputs=x, outputs=f)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#model2 = get_model2()

def model_42_1(in_len,out_len):

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(in_len,)))
    model.add(tf.keras.layers.Dense(1024, activation="selu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation="selu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation="selu"))
    model.add(tf.keras.layers.Dense(128, activation="selu"))
    model.add(tf.keras.layers.Dense(64, activation="selu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation="selu"))
    model.add(tf.keras.layers.Dense(out_len,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
    return model




    
def model_42_2(in_len,out_len):

    inp=(tf.keras.Input(shape=(in_len,)))

    model2=(tf.keras.layers.Dense(512,activation="tanh"))(inp)
    model2=(tf.keras.layers.Dense(256,activation="tanh"))(model2)
    model2=(tf.keras.layers.Dense(128,activation="tanh"))(model2)
    model2=(tf.keras.layers.Dense(64,activation="tanh"))(model2)
    model2=(tf.keras.layers.Dense(128,activation="tanh"))(model2)
    model2=(tf.keras.layers.Dense(256,activation="tanh"))(model2)
    model2=(tf.keras.layers.Dense(512,activation="tanh"))(model2)


    model1=(tf.keras.layers.Dense(1024, activation="selu"))(inp)
    model1=(tf.keras.layers.BatchNormalization())(model1)
    model1=(tf.keras.layers.Dense(512, activation="selu"))(model1)
    model1=(tf.keras.layers.Dropout(0.2))(model1)
    model1=(tf.keras.layers.Dense(256, activation="selu"))(model1)
    model1=(tf.keras.layers.Dense(128, activation="selu"))(model1)
    model1=(tf.keras.layers.Dense(64, activation="selu"))(model1)
    model1=(tf.keras.layers.Dropout(0.2))(model1)
    model1=(tf.keras.layers.Dense(32, activation="selu"))(model1)


    model_comb=tf.keras.layers.Concatenate(axis=-1)([model1 , model2])
    model_comb=(tf.keras.layers.Dense(64,activation="tanh"))(model_comb)
    model_comb=(tf.keras.layers.Dense(32,activation="tanh"))(model_comb)
    model_comb=(tf.keras.layers.Dense(16,activation="tanh"))(model_comb)


    model_out=(tf.keras.layers.Dense(out_len,activation='softmax'))(model_comb)
    model=tf.keras.Model(inputs=inp,outputs=model_out)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


model_42_1(17,10)
model_42_2(17,10)