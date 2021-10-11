# Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath('..'))

from sklearn.model_selection import train_test_split
from utils.augmentation import augment_3dimages_stacked
from utils.common import plot_training_graphs, get_classes_list, train_model
from model_architectures._3dcnn import load_model_binary

def main():
    df = pd.read_csv(r'C:\Users\Shreyansh\Desktop\Microexpression Detection\samm_data.csv')
    print(df.head())

    '''A sample image'''
    img = image.load_img(r"C:\Users\Shreyansh\Desktop\Microexpression Detection\SAMM\017\017_3_3\017_1395.jpg")
    plt.imshow(img)

    output = get_classes_list(df, "binary")
    positive = output[0]
    negative = output[1]

    print(len(positive))
    print(len(negative))
    print(positive[0])
    print(negative[0])

    '''Because of the heavy imbalance (and small size of the trainable images), it is imperative to perform augmentation.'''
    output = augment_3dimages_stacked([positive, negative], "binary")

    data = np.array(output[0])
    labels = np.array(output[1])

    print(data.shape)
    print(labels.shape)
    print(np.unique(labels, return_counts=True))

    '''Splitting the data into training and validation sets in the ratio 70:30.'''
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=2)

    print(trainX.shape)
    print(testX.shape)
    print(trainY.shape)
    print(testY.shape)

    '''Load model.'''
    model = load_model_binary()
    print(model.summary())

    '''Storing training parameters.'''
    p = dict()
    p['lr'] = 0.002
    p['loss_function'] = 'sparse_categorical_crossentropy'
    p['optimizer'] = keras.optimizers.Adam
    p['metrics'] = ['accuracy']
    p['epochs'] = 23
    p['batch_size'] = 32
    p['validation_batch_size'] = 32

    model_fit = train_model(model, trainX, trainY, testX, testY, p)

    '''Plotting graphs of training and validation (both loss and accuracy), for visualization.'''
    plot_training_graphs(model_fit)

if __name__ == '__main__':
    main()