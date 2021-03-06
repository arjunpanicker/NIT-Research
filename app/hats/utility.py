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
        top_nearest_words_to_devices[device].extend([device])

    for word in command.split(' '):
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
    '''Split the data according to the OVR mechanism for 
    per class training.
    '''
    classList = dataset['label'].unique()
    for label in classList:
        X, y = dataset['sent_vec'], dataset[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, \
            random_state=40, stratify=y)
        X_train, X_test = np.stack(X_train), np.stack(X_test)
        yield np.asarray(X_train), np.asarray(X_test), \
            np.asarray(y_train), np.asarray(y_test), label

def data_split(dataset: pd.DataFrame, test_size: float = 0.25):
    '''Split the dataset into train and test sets.
    '''
    X, y = dataset['sent_vec'], dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, \
                random_state=40, stratify=y)
    train_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_test)
    train_df['y'], test_df['y'] = y_train, y_test
    return train_df, test_df

def plot(models):
    fig = plt.figure(figsize=(20, 60))
    plot_count = 1
    for m_name in models.keys():
        history = models[m_name]['history'].history
        plt.subplot(len(models.keys()), 3, plot_count)
        plt.xlabel('epochs')
        plt.grid()
        plt.ylabel('loss')
        plt.xticks(range(0, len(history['loss']) + 1, 5))
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title(map_label(m_name))
        plt.legend(['Train Set', 'Validation Set'], loc='upper right')
        plot_count += 1
        
        plt.subplot(len(models.keys()), 3, plot_count)
        plt.xlabel('epochs')
        plt.grid()
        plt.ylabel('F1 Score')
        plt.xticks(range(0, len(history['_f1_score']) + 1, 5))
        plt.plot(history['_f1_score'])
        plt.plot(history['val__f1_score'])
        plt.title(map_label(m_name))
        plt.legend(['Train Set', 'Validation Set'], loc='lower right')
        plot_count += 1

        plt.subplot(len(models.keys()), 3, plot_count)
        plt.xlabel('epochs')
        plt.grid()
        plt.ylabel('accuracy')
        plt.xticks(range(0, len(history['accuracy']) + 1, 5))
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title(map_label(m_name))
        plt.legend(['Train Set', 'Validation Set'], loc='lower right')
        plot_count += 1

    return fig

def map_label(label:str)-> str:
    label_map = {
        '__label__light_off': 'light off',
        '__label__light_on': 'light on',
        '__label__geyser_on': 'geyser on',
        '__label__geyser_off': 'geyser off',
        '__label__fan_on': 'fan on',
        '__label__fan_off': 'fan off',
        '__label__tv_on': 'tv on',
        '__label__tv_off': 'tv off',
        '__label__ac_on': 'ac on',
        '__label__ac_off': 'ac off',
        'Other': 'other'
        }

    return label_map[label]
