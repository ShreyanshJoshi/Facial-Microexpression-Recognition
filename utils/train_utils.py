import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.callbacks import ReduceLROnPlateau

def run_kfold_cv(data, labels, model, p, mode, runs=3):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    avg_loss_across_runs = []
    avg_acc_across_runs = []
    avg_std_across_runs = []
    greatest = -1

    for _ in range(runs):
        fold_no = 1

        # Define per-fold score containers 
        acc_per_fold = []
        loss_per_fold = []

        for train, test in kfold.split(data, labels):
            (trainX, valX, trainY, valY) = train_test_split(data[train], labels[train], test_size=0.27, stratify=labels[train])
            if mode=="binary":
                trainY = np_utils.to_categorical(trainY, 2)
                valY = np_utils.to_categorical(valY, 2)
                testX = data[test]
                testY = np_utils.to_categorical(labels[test], 2)
            
            else:
                trainY = np_utils.to_categorical(trainY, 7)
                valY = np_utils.to_categorical(valY, 7)
                testX = data[test]
                testY = np_utils.to_categorical(labels[test], 7)
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            train_model(model, trainX, trainY, valX, valY, p)
            
            # Generate generalization metrics
            scores = model.evaluate(testX, testY)
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
            
            if scores[1] > greatest:
                greatest = scores[1]
                model.save('/content/drive/MyDrive/Microexpression Detection/Saved models/3D CNN_multiclass')

            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            fold_no = fold_no + 1

        # == Provide average scores once training is done ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')

        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')
        avg_acc_across_runs.append(np.mean(acc_per_fold))
        avg_loss_across_runs.append(np.mean(loss_per_fold))
        avg_std_across_runs.append(np.std(acc_per_fold))

    print('****************************************************************************************')
    print('Average scores across all K-fold runs:')
    print(f'> Accuracy: {np.mean(avg_acc_across_runs)} (+- {np.std(avg_std_across_runs)})')
    print(f'> Loss: {np.mean(avg_loss_across_runs)}')


'''Compiles, trains the model with the given hyperparameters.'''
def train_model(model, trainX, trainY, valX, valY, p, model_name="3dcnn"):
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)
    model.compile(optimizer = p['optimizer'](learning_rate=p['lr']), loss=p['loss_function'], metrics=p['metrics'])

    # Training the model
    history = model.fit(trainX, trainY, epochs=p['epochs'], batch_size=p['batch_size'], steps_per_epoch=len(trainX)//p['batch_size'], 
    validation_data=(valX, valY), validation_batch_size=p['validation_batch_size'], validation_steps=len(valX)//p['validation_batch_size'],
    callbacks=[reduce_lr])

    if model_name=="2dcnn":
        return history