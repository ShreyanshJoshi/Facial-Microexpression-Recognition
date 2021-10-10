# Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from utils.common import plot_training_graphs, get_classes_list, train_model
from models._2dcnn import load_model_binary
from utils.augmentation import augment_2dimages_open
from utils.miscellaneous import get_classes_list_with_9frames_and_labels_2dimages_open, get_train_test_list_2dimages_open, test_predictions

def main():
    df = pd.read_csv(r'C:\Users\Shreyansh\Desktop\Microexpression Detection\samm_data.csv')
    print(df.head())

    # A sample image
    img = image.load_img(r"C:\Users\Shreyansh\Desktop\Microexpression Detection\SAMM\017\017_3_3\017_1395.jpg")
    plt.imshow(img)

    output = get_classes_list(df, "binary")
    positive = output[0]
    negative = output[1]

    print(len(positive))
    print(len(negative))

    # Printing a sample element of the returned list, to check the format
    print(positive[0])

    output = get_classes_list_with_9frames_and_labels_2dimages_open([positive, negative], "binary")
    positive1 = output[0]
    negative1 = output[1]

    print(len(positive1))
    print(len(negative1))
    print(np.array(positive1).shape)
    print(np.array(negative1).shape)

    # Printing a sample element of the returned list, to check the format
    print(positive1[0])
    
    output = get_train_test_list_2dimages_open([positive1, negative1], "binary")
    train_list = output[0]
    test_list = output[1]
    print(len(train_list))
    print(len(test_list))

    # Augmentation doesn't add value in this case as binary classification is intrinsically a not so difficult task and secondly, because we are treating each of the 9 frames of a datapoint individually in this case, we have 9 times the datapoints available for training (as compared to the cases when we stacked the 9 frames depthwise to produce a single 3D image). Adding augmentation only introduces redundancy in the dataset.
    output = augment_2dimages_open(train_list, "binary", augment=False)
    data = np.array(output[0])
    labels = np.array(output[1])

    print(data.shape)
    print(labels.shape)
    print(np.unique(labels, return_counts=True))

    # Splitting the data into training and validation sets in the ratio 75:25
    (trainX, valX, trainY, valY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=2)

    print(trainX.shape)
    print(valX.shape)
    print(trainY.shape)
    print(valY.shape)

    # Load model architecture
    model = load_model_binary()
    print(model.summary())

    # Storing training parameters
    p = dict()
    p['lr'] = 0.0035
    p['loss_function'] = 'sparse_categorical_crossentropy'
    p['optimizer'] = keras.optimizers.Adam
    p['metrics'] = ['accuracy']
    p['epochs'] = 25
    p['batch_size'] = 128
    p['validation_batch_size'] = 64

    model_fit = train_model(model, trainX, trainY, valX, valY, p)

    plot_training_graphs(model_fit)

    # Making predictions and finding test accuracy on the test set
    test_predictions(test_list, model, "binary")

if __name__ == '__main__':
    main()