# Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
from keras.utils import np_utils 
from sklearn.model_selection import train_test_split
from utils.augmentation import augment_3dimages_stacked
from utils.common import plot_training_graphs, get_classes_list, train_model
from models._3dcnn import load_model_multiclass

def main():
    df = pd.read_csv(r'C:\Users\Shreyansh\Desktop\Microexpression Detection\samm_data.csv')
    print(df.head())

    '''A sample image'''
    img = image.load_img(r"C:\Users\Shreyansh\Desktop\Microexpression Detection\SAMM\017\017_3_3\017_1395.jpg")
    plt.imshow(img)

    output = get_classes_list(df, "multiclass")
    happiness = output[0]
    surprise = output[1]
    anger = output[2]
    fear = output[3]
    disgust = output[4]
    contempt = output[5]
    sadness = output[6]

    print(len(happiness), ' ', len(surprise), ' ', len(anger), ' ', len(fear), ' ', len(disgust), ' ', len(contempt), ' ', len(sadness))
    
    '''Because of the heavy imbalance (and small size of the dataset), it is imperative to perform augmentation.'''
    output = augment_3dimages_stacked([happiness, surprise, anger, fear, disgust, contempt, sadness], "multiclass")
    
    data = np.array(output[0])
    labels = np.array(output[1])

    print(data.shape)
    print(labels.shape)
    print(np.unique(labels, return_counts=True))

    labels = np_utils.to_categorical(labels, 7)         # OHE is essential in multi-class to avoid any unwarranted assumptions between class numbers
    
    '''Splitting the data into training and validation sets in the ratio 80:20.'''
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=2)

    print(trainX.shape)
    print(testX.shape)
    print(trainY.shape)
    print(testY.shape)

    '''Load model.'''
    model = load_model_multiclass()
    print(model.summary())

    '''Storing training parameters.'''
    p = dict()
    p['lr'] = 0.004
    p['loss_function'] = 'categorical_crossentropy'
    p['optimizer'] = keras.optimizers.Adam
    p['metrics'] = ['accuracy']
    p['epochs'] = 40
    p['batch_size'] = 32
    p['validation_batch_size'] = 32

    model_fit = train_model(model, trainX, trainY, testX, testY, p)

    '''Plotting graphs of training and validation (both loss and accuracy), for visualization.'''
    plot_training_graphs(model_fit)


if __name__ == '__main__':
    main()