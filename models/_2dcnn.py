import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

'''Binary classification, 3D CNN architecture that gave the best results. Because in case of 2D CNN (majority voting policy), we are actually using each of the 9 
frames as an individual datapoint, the number of datapoints for this task are quite more than in the cases when we concatenate the 9 frames depthwise and use them 
together as a single image. Consequently, because of a resonable number of images to train, models tend to overfit less. Hence a deeper model was used as compared
to it's 3D CNN counterpart. '''
def load_model_binary():
    inputs = keras.Input((128, 128, 1))

    x = Conv2D(filters=32, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform')(inputs)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dense(units=64, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    outputs = Dense(units=2, activation="softmax", kernel_initializer='he_uniform')(x)
    model = keras.Model(inputs, outputs, name="2dcnn")
    return model

'''Multiclass classification is much more complex than binary classification, and even more so with the limited data we have at our disposal. Hence, a deeper network 
is required to learn the underlying patterns properly (especially with 2D CNNs that can't fathom the temporal relations between frames) '''
def load_model_multiclass():
    inputs = keras.Input((128, 128, 1))

    x = Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform')(inputs)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dense(units=384, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(0.42)(x)
    x = BatchNormalization()(x)

    x = Dense(units=96, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)

    outputs = Dense(units=7, activation="softmax", kernel_initializer='he_uniform')(x)
    model = keras.Model(inputs, outputs, name="2dcnn")
    return model