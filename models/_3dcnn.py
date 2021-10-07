import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from keras.layers import Conv3D, MaxPool3D

# Binary classification, 3D CNN architecture that gave the best results. B'se of the extremely small size of the dataset, even after augmentation, and perhaps the simplicity associated with binary classification, deeper models were found to overfit very easily (predicting all as negative- bias due to class imbalance). To mitigate that, a very shallow architecture has been used, along with intensive regularization in the form of dropouts and batchnorm.
def load_model_binary():
    inputs = keras.Input((128, 128, 9, 1))

    x = Conv3D(filters=32, kernel_size=(3,3,3), activation="relu", kernel_initializer='he_uniform')(inputs)
    x = MaxPool3D(pool_size=(2,2,1))(x)

    x = Flatten()(x)

    x = Dense(units=32, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(0.42)(x)
    x = BatchNormalization()(x)

    outputs = Dense(units=2, activation="softmax", kernel_initializer='he_uniform')(x)
    model = keras.Model(inputs, outputs, name="3dcnn_binary")
    return model


# Multiclass classification, 3D CNN architecture that gave the best results. Since, multiclass classification is intrinsically more complicated than binary classification, deeper models have been used to understand & learn the complex underlying patterns. 
def load_model_multiclass():
    inputs = keras.Input((128, 128, 9, 1))

    x = Conv3D(filters=64, kernel_size=(3,3,2), activation="relu", kernel_initializer='he_uniform')(inputs)
    x = MaxPool3D(pool_size=(2,2,2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=(3,3,2), activation="relu", kernel_initializer='he_uniform')(x)
    x = MaxPool3D(pool_size=(2,2,2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dense(units=256, activation="relu", kernel_initializer='he_uniform', kernel_regularizer=regularizers.l1_l2(l1=0.02, l2=0.02))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)

    x = Dense(units=64, activation="relu", kernel_initializer='he_uniform', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.27)(x)
    x = BatchNormalization()(x)

    outputs = Dense(units=7, activation="softmax", kernel_initializer='he_uniform')(x)
    model = keras.Model(inputs, outputs, name="3dcnn_multiclass")
    return model