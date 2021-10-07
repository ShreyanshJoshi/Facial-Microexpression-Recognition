# Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import random
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from utils.common import plot_training_graphs, get_classes_list
from models._2dcnn import load_model_multiclass
from utils.miscellaneous import get_classes_list_with_9frames_and_labels_2dcnn, get_train_test_list_2dcnn, test_predictions
from utils.augmentation import augment_2dimages_open

def main():
    df = pd.read_csv(r'C:\Users\Shreyansh\Desktop\Microexpression Detection\samm_data.csv')
    print(df.head())

    # A sample image
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

    output = get_classes_list_with_9frames_and_labels_2dcnn([happiness, surprise, anger, fear, disgust, contempt, sadness], "multiclass")
    happiness1 = output[0]
    surprise1 = output[1]
    anger1 = output[2]
    fear1 = output[3]
    disgust1 = output[4]
    contempt1 = output[5]
    sadness1 = output[6]

    # Printing a sample element of the returned list for checking the format
    print(np.array(anger1).shape)
    print(anger1[0])

    output = get_train_test_list_2dcnn([happiness1, surprise1, anger1, fear1, disgust1, contempt1, sadness1], "multiclass")
    train_list = output[0]
    test_list = output[1]
    print(len(train_list))
    print(len(test_list))

    # Augmentation (of training dataset) is required in case of multiclass as the distribution of images in different classes is very diverse and imbalanced
    output = augment_2dimages_open(train_list, "multiclass", augment=True)
    data = np.array(output[0])
    labels = np.array(output[1])

    print(data.shape)
    print(labels.shape)
    print(np.unique(labels, return_counts=True))

    labels = np_utils.to_categorical(labels, 7)

    # Splitting the data into training and validation sets in the ratio 75:25
    (trainX, valX, trainY, valY) = train_test_split(data, labels, test_size=0.25, shuffle=True, stratify=labels, random_state=2)

    print(trainX.shape)
    print(valX.shape)
    print(trainY.shape)
    print(valY.shape)

    # Load model
    model = load_model_multiclass()
    print(model.summary())

    # Prepare for training
    initial_learning_rate = 0.004
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.8, staircase=True
    )
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics =['accuracy'])

    # Training the model
    model_fit = model.fit(trainX, trainY, epochs=1, batch_size=128, steps_per_epoch=len(trainX)//128, validation_data=(valX, valY), 
                        validation_batch_size=64, validation_steps=len(valX)//64)
    plot_training_graphs(model_fit)

    # Making predictions and finding test accuracy
    test_predictions(test_list, model, "multiclass")


if __name__ == '__main__':
    main()

