# Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import sys, os
import argparse
sys.path.append(os.path.abspath('..'))

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from utils.common import plot_training_graphs, get_classes_list
from utils.train_utils import train_model
from utils.augmentation import augment_2dimages
from utils.miscellaneous import get_classes_list_with_9frames_and_labels, get_train_test_list, test_predictions_2dcnn

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', help='Binary or multiclass classification')
FLAGS.add_argument('--model', help='Model to be executed')

def main():
    args = FLAGS.parse_args()
    if args.mode is None or args.model is None:
        raise ValueError("Missing argument(s).")

    if args.model!="2dcnn":
        raise ValueError("Invalid model name entered.")
        
    df = pd.read_csv(r'C:\Users\Shreyansh\Desktop\Microexpression Detection\samm_data.csv')
    print(df.head())

    '''A sample image.'''
    img = image.load_img(r"C:\Users\Shreyansh\Desktop\Microexpression Detection\SAMM\017\017_3_3\017_1395.jpg")
    plt.imshow(img)

    if args.mode=="binary":
        from models.binary_models import load_model_2dcnn
        output = get_classes_list(df, "binary")
        positive = output[0]
        negative = output[1]
        print(len(positive), ' ', len(negative))

        output = get_classes_list_with_9frames_and_labels([positive, negative], "binary")
        output = get_train_test_list([output[0], output[1]], "binary")
    
    elif args.mode=="multiclass":
        from models.multiclass_models import load_model_2dcnn
        output = get_classes_list(df, "multiclass")
        happiness = output[0]
        surprise = output[1]
        anger = output[2]
        fear = output[3]
        disgust = output[4]
        contempt = output[5]
        sadness = output[6]

        print(len(happiness), ' ', len(surprise), ' ', len(anger), ' ', len(fear), ' ', len(disgust), ' ', len(contempt), ' ', len(sadness))

        output = get_classes_list_with_9frames_and_labels([happiness, surprise, anger, fear, disgust, contempt, sadness], "multiclass")
        output = get_train_test_list([output[0], output[1], output[2], output[3], output[4], output[5], output[6]], "multiclass")
    
    else:
        raise ValueError("Invalid mode entered.")

    train_list = output[0]
    test_list = output[1]
    print(len(train_list))
    print(len(test_list))

    output = augment_2dimages(train_list, args.mode, augment=False if args.mode=="binary" else True)

    data = np.array(output[0])
    labels = np.array(output[1])
    print(data.shape)
    print(labels.shape)
    print(np.unique(labels, return_counts=True))

    if args.mode=="binary":
        labels = np_utils.to_categorical(labels, 2)
    
    else:
        labels = np_utils.to_categorical(labels, 7)

    '''Splitting the data into training and validation sets in the ratio 75:25.'''
    (trainX, valX, trainY, valY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=2)

    print(trainX.shape)
    print(valX.shape)
    print(trainY.shape)
    print(valY.shape)

    '''Load model architecture'''
    model = load_model_2dcnn()
    print(model.summary())

    '''Storing training parameters'''
    p = dict()
    p['lr'] = 0.0035
    p['loss_function'] = 'categorical_crossentropy'
    p['optimizer'] = keras.optimizers.Adam
    p['metrics'] = ['accuracy']
    p['epochs'] = 25
    p['batch_size'] = 128
    p['validation_batch_size'] = 64

    model_fit = train_model(model, trainX, trainY, valX, valY, p, args.model)

    '''Plotting graphs of training and validation (both loss and accuracy), for visualization.'''
    plot_training_graphs(model_fit)

    '''Making predictions and finding accuracy on the test set.'''
    test_predictions_2dcnn(test_list, model, args.mode)

if __name__ == '__main__':
    main()