import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from .config import CONFIG
# Types of devices in the model
_deviceList = ['ac', 'tv', 'fan', 'light', 'geyser']

def device_exists(command: str, ft_model) -> bool:
    '''Checks whether the given string contains one of the many devices
    supported by the model. 
    '''
    # Get the 4 nearest words to each device and create a dictionary
    top_nearest_words_to_devices = {}
    for device in _deviceList:
        top_nearest_words_to_devices[device] = []
        nearestWords = ft_model.get_nearest_neighbors(device, k=4)
        top_nearest_words_to_devices[device].extend([word for _, word in nearestWords])

    for word in command:
        for device in _deviceList:
            if word in top_nearest_words_to_devices[device]:
                return True
    
    return False

def add_class_ovr_cols(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Add a column for each class representing whether that class
    is present for that instance or not (OVR Technique)
    '''
    classList = dataset['label'].unique()
    for label in classList:
        dataset[label] = np.where(dataset['label'] == label, 1, 0)
    return dataset

def shuffle_split(dataset: pd.DataFrame, label: str):
    '''A generator function to split the dataset using 
    StratifiedShuffleSplit and return each split
    '''
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=20)

    # Code taken from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    X, y = dataset['sent_vec'], dataset[label]
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = np.stack(X.iloc[train_index]), np.stack(X.iloc[test_index])
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        yield np.asarray(X_train), np.asarray(X_test), \
            np.asarray(y_train), np.asarray(y_test)

def data_split_classwise(dataset: pd.DataFrame):
    classList = dataset['label'].unique()
    for label in classList:
        X, y = dataset['sent_vec'], dataset[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, \
            random_state=40, stratify=y)
        X_train, X_test = np.stack(X_train), np.stack(X_test)
        yield np.asarray(X_train), np.asarray(X_test), \
            np.asarray(y_train), np.asarray(y_test), label

def data_split(dataset: pd.DataFrame):
    X, y = dataset['sent_vec'], dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, \
                random_state=40, stratify=y)
    X_train, X_test = np.stack(X_train), np.stack(X_test)
    return X_train, X_test, y_train, y_test

def plot(modelHistory, titleName, filename: str):
    fig = plt.figure(figsize=(7, 5), dpi=300)
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(modelHistory['loss'])
    ax1.plot(modelHistory['val_loss'])
    fig.suptitle(titleName)
    ax1.set_ylabel('loss')

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(modelHistory['accuracy'])
    ax2.plot(modelHistory['val_accuracy'])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend(['train', 'validation'], loc='upper right')
    plt.gcf().subplots_adjust(bottom=0.25, left=0.25)
    # extent = full_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('..' + CONFIG.OUTPUT_DIRECTORY_NAME + \
        CONFIG.PLOT_DIRECTORY_NAME + filename)