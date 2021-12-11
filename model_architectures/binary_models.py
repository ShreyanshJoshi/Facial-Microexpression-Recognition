import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, ConvLSTM2D, MaxPool3D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential 

def load_model_2dcnn():
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


def load_model_3dcnn():
    inputs = keras.Input((9, 128, 128, 1))

    x = Conv3D(filters=64, kernel_size=(1,3,3), activation="relu", kernel_initializer='he_uniform')(inputs)
    x = MaxPool3D(pool_size=(1,2,2))(x)

    x = Conv3D(filters=96, kernel_size=(1,3,3), activation="relu", kernel_initializer='he_uniform')(x)
    x = MaxPool3D(pool_size=(1,2,2))(x)

    x = Flatten()(x)

    x = Dense(units=24, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(0.35)(x)

    outputs = Dense(units=2, activation="softmax", kernel_initializer='he_uniform')(x)
    model = keras.Model(inputs, outputs, name="3dcnn_binary")
    return model

def load_model_cnn_lstm():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(9, 128, 128, 1)))
    model.add(TimeDistributed(MaxPooling2D((2,2))))

    model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))

    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(.3))

    model.add(LSTM(32, activation='relu', dropout=0.4, return_sequences=False))

    model.add(Dense(24, activation='relu'))
    model.add(Dropout(.35))

    model.add(Dense(2, activation='softmax'))

    return model

def load_model_convlstm2d():
    model = Sequential()
    model.add(ConvLSTM2D(32, kernel_size=(3, 3), strides=(1,1), return_sequences=True, activation='relu', input_shape=(9, 128, 128, 1)))
    model.add(MaxPooling3D((1,2,2)))

    model.add(Flatten())

    model.add(Dense(48, activation='relu'))
    model.add(Dropout(.3))

    model.add(Dense(2, activation='softmax'))

    return model