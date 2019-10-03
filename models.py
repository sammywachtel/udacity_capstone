# Initial model inspired by https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
import pandas as pd
import numpy as np
import tensorflow as tf

# Convolutional layers
#   Input shape: (samples, channels, rows, cols)
#   Output shape: (samples, filters, new_rows, new_cols)
    

## V1 results (UNTRAINED) with
### test shape (4096, 3)
## Loss:  2.3066686056554317  Accuracy:  0.11987305
## Total test records: 4096
## Number of correct: 451
## Percent correct: 0.110107421875

## V1 results (weights.best.v1_1.hdf5) with:
### train shape (10000, 3)
### valid shape (430, 3)
### test shape (4096, 3)
## Loss:  1.0408632545731962  Accuracy:  0.6882324
## Total test records: 4096
## Number of correct: 553
## Percent correct: 0.135009765625
######
## V1 results (weights.best.v1_2.hdf5) with:
## Loss:  1.621183446983793  Accuracy:  0.65998137
### train shape (50000, 3)
### valid shape (2150, 3)
### test shape (4096, 3)
## Loss:  0.9963033711537719  Accuracy:  0.69970703
## Total test records: 4096
## Number of correct: 552
## Percent correct: 0.134765625
#####NOTES: No improvement with larger training set
def create_model_v1(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    
    if show_summary:
        model.summary()
    
    return model

## V2 results (weights.best.v1_1.hdf5) with:
## Loss:  1.5549413437273965  Accuracy:  0.7168843
### train shape (50000, 3)
### valid shape (2150, 3)
### test shape (4096, 3)
## Loss:  0.9504697730299085  Accuracy:  0.736084
## Total test records: 4096
## Number of correct: 527
## Percent correct: 0.128662109375
######

def create_model_v2(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,
                                  epsilon=1e-07),loss="categorical_crossentropy",metrics=["accuracy"])
    
    if show_summary:
        model.summary()
    
    return model

## V3 results (UNTRAINED) with
### test shape (4096, 3)
## Loss:  1.0019865212962031  Accuracy:  0.13134766
## Total test records: 4096
## Number of correct: 601
## Percent correct: 0.146728515625

## V3 results (weights.best.v3_1.hdf5) with:
## Loss:  1.5549413437273965  Accuracy:  0.7168843
### train shape (50000, 3)
### valid shape (2150, 3)
### test shape (4096, 3)
## Loss:  1.0000212732702494  Accuracy:  0.20581055
## Total test records: 4096
## Number of correct: 843
## Percent correct: 0.205810546875
######

def create_model_v3(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,
                                  epsilon=1e-07),loss="categorical_hinge",metrics=["accuracy"])
    #sparse_categorical_crossentropy 
    if show_summary:
        model.summary()
    
    return model


## V4 results (UNTRAINED) with
### test shape (4096, 3)
## Loss:  1.0029726549983025  Accuracy:  0.07055664
## Total test records: 4096
## Number of correct: 311
## Percent correct: 0.075927734375

## V4 results (weights.best.v4_1.hdf5) with:
## Loss:  1.000329845584929  Accuracy:  0.122558594
### train shape (50000, 3)
### valid shape (2150, 3)
### test shape (4096, 3)
## Total test records: 4096
## Number of correct: 502
## Percent correct: 0.12255859375
######
def create_model_v4(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,
                                  epsilon=1e-07),loss="categorical_hinge",metrics=["accuracy"])
    #sparse_categorical_crossentropy 
    if show_summary:
        model.summary()
    
    return model

## V5 results (UNTRAINED) with
### test shape (4096, 3)
## Loss:  1.0029314439743757  Accuracy:  0.0793457
## Total test records: 4096
## Number of correct: 318
## Percent correct: 0.07763671875

## V5 results (weights.best.v5_1.hdf5) with:
## Loss:  1.0000042061307537  Accuracy:  0.19402985
### train shape (50000, 3)
### valid shape (2150, 3)
### test shape (4096, 3)
## Loss:  1.0000026049092412  Accuracy:  0.18701172
## Total test records: 4096
## Number of correct: 766
## Percent correct: 0.18701171875
######
def create_model_v5(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999,
                                  epsilon=1e-07),loss="categorical_hinge",metrics=["accuracy"])
    #sparse_categorical_crossentropy 
    if show_summary:
        model.summary()
    
    return model

## V6 results (UNTRAINED) with
## didn't work at all
######
def create_model_v6(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.Adamax(),loss="categorical_hinge",metrics=["accuracy"])
    #sparse_categorical_crossentropy 
    if show_summary:
        model.summary()
    
    return model

## V7 results (UNTRAINED) with
## Loss:  1.0031691053882241  Accuracy:  0.037597656
## Total test records: 4096
## Number of correct: 226
## Percent correct: 0.05517578125

## V7 results (weights.best.v7_1.hdf5) with:
### train shape (283704, 3)
### valid shape (12678, 3)
### test shape (4096, 3)
## Loss:  1.0000030007213354  Accuracy:  0.034423828
## Total test records: 4096
## Number of correct: 141
## Percent correct: 0.034423828125
######
def create_model_v7(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999,
                                  epsilon=1e-07),loss="categorical_hinge",metrics=["accuracy"])
    #sparse_categorical_crossentropy 
    if show_summary:
        model.summary()
    
    return model

def create_model_v8(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(256,256,3)))
    model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Conv2D(256, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    if show_summary:
        model.summary()
    
    return model

def create_model_v9(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,
                                  epsilon=1e-07),loss="categorical_hinge",metrics=["accuracy"])
    if show_summary:
        model.summary()
    
    return model

# The dense layer output was reduced to 8 because we started using a reduced dataset (only acoustic within the range of C4 and C5). The number of bass and organ samples was too low and have been removed. 
## NOTE: Always choses the same instrument class!
def create_model_v10(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizers.RMSprop(lr=0.005, decay=1e-6), loss="categorical_crossentropy",
                  metrics=["accuracy"])

    if show_summary:
        model.summary()
    
    return model

# seems to overfit.
# Categorical Accuracy = .7
def create_model_v11(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same', input_shape=(16,16,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (8, 8)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-5), loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    
    if show_summary:
        model.summary()
    
    return model

def create_model_v12(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same', input_shape=(16,16,3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (8, 8)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-4), loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    
    if show_summary:
        model.summary()
    
    return model

# Loading checkpoint:  saved_models/weights.best.v13_1.hdf5
## Loss:  0.9755629178355721  Categorical Accuracy:  0.7147059
## Total test records: 340
## Number of correct: 243
## Percent correct: 0.7147058823529412
def create_model_v13(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same', input_shape=(16,16,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (8, 8)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-5), loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    
    if show_summary:
        model.summary()
    
    return model

# Loading checkpoint:  saved_models/weights.best.v14_1.hdf5
## Loss:  1.0673524526988758  Categorical Accuracy:  0.6647059
## Total test records: 340
## Number of correct: 226
## Percent correct: 0.6647058823529411
def create_model_v14(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same', input_shape=(16,16,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (8, 8)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-5), loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    
    if show_summary:
        model.summary()
    
    return model

# Loading checkpoint:  saved_models/weights.best.v15_1.hdf5
## Loss:  0.9298041936229257  Categorical Accuracy:  0.7352941
## Total test records: 340
## Number of correct: 250
## Percent correct: 0.7352941176470589
def create_model_v15(show_summary=False):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same', input_shape=(21,20,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (8, 8)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-5), loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    
    if show_summary:
        model.summary()
    
    return model

def create_output_model(model, layer_indexes, show_summary=False):
    outs = [model.layers[cnt].output for cnt in layer_indexes]
    print('output layers')
    [print('-', out) for out in outs]
    ret_mod = Model(inputs=model.inputs, outputs=outs)
    if show_summary:
        ret_mod.summary()
    return ret_mod
