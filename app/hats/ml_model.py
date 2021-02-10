import fasttext
import os
import numpy as np
import pickle
from tqdm import tqdm

# Tensorflow
import tensorflow as tf
from tensorflow import keras

# Scikit-learn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .config import *
from . import utility as ut

def createFasttextModel(filename: str):
    path: str = os.path.join("../", CONFIG.OUTPUT_DIRECTORY_NAME)
    ft_model = fasttext.train_supervised(os.path.join(path, filename), \
                                dim=CONFIG.FT_DIMS, lr=0.5, epoch=40, verbose=1)
    ft_model.save_model(os.path.join(path, 'ft_model.ftz'))
    return ft_model

def modelPredict(command: str) -> str:
    # TODO: Complete this function
    pass

def predict(command: str, ft_model, filename: str) -> str:
    devicePresent: bool = ut.device_exists(command, ft_model)

    if devicePresent:
        if filename:
            loaded_model = pickle.load(open(filename, 'rb'))
            command_vec = np.reshape(ft_model.get_sentence_vector(command), (1, -1))
            result_proba = np.max(loaded_model.predict_proba(command_vec)[0])

            if result_proba > CONFIG.THRESHOLD:
                return loaded_model.classes_[np.argmax(loaded_model.predict_proba(command_vec)[0])]
            else:
                return 'Other'
    else:
        return 'Other'

def createPerceptronModels(model_names: list):
    ''' Create a perceptron for the number of models (model names) passed as arguments
    '''
    modelDict = dict()
    for name in model_names:
        model = keras.Sequential([
                    keras.layers.Dense(units=1, input_shape=(150,), \
                        kernel_initializer=keras.initializers.GlorotNormal(), activation='sigmoid')
                ])
        modelDict[name] = model

    return modelDict

def train(X_train, y_train):
    '''Creates and trains the models with the data passed as arguments
    '''
    classifierList = [
        {
            'model_name': SVC.__name__,
            'model': SVC(probability=True, random_state=40),
            'parameters': dict(C = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10e1, 10e2, 10e4], 
                kernel=['linear', 'rbf', 'poly']),
            'filename': '../' + CONFIG.OUTPUT_DIRECTORY_NAME + CONFIG.SVM_MODEL_SAVEFILE
        },
        {   
            'model_name': LogisticRegression.__name__,
            'model': LogisticRegression(random_state=40),
            'parameters': dict(C = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10e1, 10e2, 10e4],
                multi_class = ['ovr', 'multinomial'],
                solver=['liblinear', 'newton-cg', 'sag', 'saga', 'libfgs']),
            'filename': '../' + CONFIG.OUTPUT_DIRECTORY_NAME + CONFIG.SVM_MODEL_SAVEFILE
        }
    ]

    classifiers = {}

    for clfDetail in tqdm(classifierList):
        clf = GridSearchCV(estimator=clfDetail['model'], param_grid=clfDetail['parameters'])
        clf.fit(X_train, y_train)
        
        # Save the classifier
        pickle.dump(clf, open(clfDetail['filename'], 'wb'))
        classifiers[clfDetail['model_name']] = {
            'model': clf,
            'best_estimators': clf.best_estimator_,
            'filename': clfDetail['filename']
        }

        print(f"\nModel: {clfDetail['model_name']}, Train Accuracy: {clf.score(X_train, y_train)}")

    return classifiers


