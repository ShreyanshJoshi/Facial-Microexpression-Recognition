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

from utils.train_utils import run_kfold_cv
from utils.augmentation import augment_3dimages
from utils.common import get_classes_list

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', help='Binary or multiclass classification')
FLAGS.add_argument('--model', help='Model to be executed')

def main():
    args = FLAGS.parse_args()
    if args.mode is None or args.model is None:
        raise ValueError("Missing argument(s).")

    df = pd.read_csv(r'C:\Users\Shreyansh\Desktop\Microexpression Detection\samm_data.csv')
    print(df.head())

    '''A sample image'''
    img = image.load_img(r"C:\Users\Shreyansh\Desktop\Microexpression Detection\SAMM\017\017_3_3\017_1395.jpg")
    plt.imshow(img)
    
    if args.mode=="binary":
        from models.binary_models import load_model_3dcnn, load_model_cnn_lstm, load_model_convlstm2d

        output = get_classes_list(df, "binary")
        positive = output[0]
        negative = output[1]
        print(len(positive))
        print(len(negative))

        '''Because of the heavy imbalance (and small size of the trainable images), it is imperative to perform augmentation.'''
        output = augment_3dimages([positive, negative], "binary")

    elif args.mode=="multiclass":
        from models.multiclass_models import load_model_3dcnn, load_model_cnn_lstm, load_model_convlstm2d

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
        output = augment_3dimages([happiness, surprise, anger, fear, disgust, contempt, sadness], "multiclass")

    else:
        raise ValueError("Invalid mode entered.")

    data = np.array(output[0])
    labels = np.array(output[1])

    print(data.shape)
    print(labels.shape)
    print(np.unique(labels, return_counts=True))

    '''Load model.'''
    if args.model=="3dcnn":
        model = load_model_3dcnn()

    elif args.model=="cnn+lstm":
        model = load_model_cnn_lstm()
    
    elif args.model=="convlstm2d":
        model = load_model_convlstm2d()
    
    else:
        raise ValueError("Invalid model name entered.")

    print(model.summary())

    '''Storing training parameters. These are common across all models typically.'''
    p = dict()
    p['lr'] = 0.003
    p['loss_function'] = 'categorical_crossentropy'
    p['optimizer'] = keras.optimizers.Adam
    p['metrics'] = ['accuracy']
    p['epochs'] = 25
    p['batch_size'] = 16
    p['validation_batch_size'] = 16

    '''Running K-fold Cross validation 3 times'''
    run_kfold_cv(data, labels, model, p, args.mode, 3)

if __name__ == '__main__':
    main()