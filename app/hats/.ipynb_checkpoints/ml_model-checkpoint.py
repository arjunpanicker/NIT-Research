import fasttext
import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import sem
from prettytable import PrettyTable

# Tensorflow
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
from kerastuner.tuners import Hyperband

# Scikit-learn
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from .config import *
from . import utility as ut
from .data_preprocessing import Preprocessing

def createFasttextModel(filename: str):
    path: str = os.path.join("../", CONFIG.OUTPUT_DIRECTORY_NAME)
    ft_model = fasttext.train_supervised(os.path.join(path, filename), \
                                dim=CONFIG.FT_DIMS, lr=0.5, epoch=10, verbose=1)
    ft_model.save_model(os.path.join(path, 'ft_model.ftz'))
    return ft_model

def _recall(y_true, y_pred):
    y_true = np.ones_like(y_true) 
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    all_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + np.epsilon())
    return recall

def _precision(y_true, y_pred):
    y_true = np.ones_like(y_true) 
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + np.epsilon())
    return precision

def _f1_score(y_true, y_pred):
    precision = _precision(y_true, y_pred)
    recall = _recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+np.epsilon()))

def nn_tune_train(data: pd.DataFrame, model_names: list)->dict:
    num_classes = len(data.label.unique())

    with tqdm(total=num_classes) as bar:
        for X_train, X_test, y_train, y_test, label in ut.data_split_classwise(data):
            bar.set_description(f'Tuning on model {label}')
            for name in model_names:
                tuner = Hyperband(
                    _build_model, 
                    objective='val_loss', 
                    max_epochs=50, 
                    factor=3, 
                    directory='hyperband', 
                    project_name=f'slp{label}'
                )

                tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                print(f"{name} optimal params: {best_hps}")
            bar.update(1)

def _build_model(hp):
    kernel_initializer_list = [
        'glorot_uniform',
        'glorot_normal',
        'he_normal',
        'he_uniform'
        ]

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_kernel_init = hp.Choice('kernel_initializer', values=kernel_initializer_list, \
            default='glorot_normal')
    
    model = keras.Sequential([
                keras.layers.Dense(units=1, input_shape=(150,), \
                    kernel_initializer=hp_kernel_init, activation='sigmoid')
            ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), \
        loss='binary_crossentropy', metrics=['accuracy', _f1_score])

    return model


def nn_modelTrain(data: pd.DataFrame, models: dict)-> dict:
    '''Trains the models passed to the function on the dataframe provided.
    '''
    num_classes = len(data.label.unique())

    with tqdm(total=num_classes) as bar:
        for X_train, X_test, y_train, y_test, label in ut.data_split_classwise(data):
            bar.set_description(f'Working on model {label}')
            models[label]['history'] = models[label]['model'].fit(X_train, y_train, batch_size=20, \
                                                                epochs=50, validation_data=(X_test, y_test), \
                                                                verbose=0)
            bar.update(1)

    print("Saving models to disk...")
    for m_name in models.keys():
        models[m_name]['model'].save(f"../{CONFIG.NN_OUTPUT_DIRECTORY_NAME}{m_name}")

    return models

def nn_modelPredict(command: str, ft_model, preprocess_obj: Preprocessing, models=None, model_names=None) -> str:
    test_command_preprocessed = preprocess_obj.strpreprocessing(command)
    test_command_vec = ft_model.get_sentence_vector(test_command_preprocessed)
    test_command_vec = np.reshape(test_command_vec, (1, -1))
    print(test_command_preprocessed)

    table = PrettyTable()
    table.field_names = ['Model Name', 'Predicted Probability']
    proba_list = []
    models_list = model_names if models is None else list(models.keys())

    models = dict() if models is None else models
    if len(models) == 0:
        for name in model_names:
            models.setdefault(name, {'model': None})
            models[name]['model'] = keras.models.load_model(f"../{CONFIG.NN_OUTPUT_DIRECTORY_NAME}{name}")
    
    for m_name in models_list:
        prediction_proba = models[m_name]['model'].predict(test_command_vec)
        proba_list.extend(prediction_proba[0])
        table.add_row([ut.map_label(m_name), prediction_proba[0][0]])

    final_prediction: str = models_list[np.argmax(proba_list)] if np.max(proba_list) > CONFIG.THRESHOLD else "Other"

    print(table)
    print('\nFinal Prediction: ', final_prediction)

    return final_prediction

def predict(command: str, ft_model, filename: str) -> str:
    # devicePresent: bool = ut.device_exists(command, ft_model)

    # if devicePresent:
    if filename:
        loaded_model = pickle.load(open(filename, 'rb'))
        command_vec = np.reshape(ft_model.get_sentence_vector(command), (1, -1))
        result_proba = np.max(loaded_model.predict_proba(command_vec)[0])
        if result_proba > CONFIG.THRESHOLD:
            return loaded_model.classes_[np.argmax(loaded_model.predict_proba(command_vec)[0])]
        else:
            return 'Other'
    # else:
    #     return 'Other'

def createPerceptronModels(model_names: list):
    ''' Create a perceptron for the number of models (model names or classes) passed as arguments
    '''
    modelDict = dict()
    for name in model_names:
        model = keras.Sequential([
                    keras.layers.Dense(units=1, input_shape=(150,), \
                        kernel_initializer=keras.initializers.GlorotNormal(), activation='sigmoid')
                ])
        modelDict.setdefault(name, {'model': None})
        modelDict[name]['model'] = model

    return modelDict

def train(train_df: pd.DataFrame):
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
                solver=['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs']),
            'filename': '../' + CONFIG.OUTPUT_DIRECTORY_NAME + CONFIG.LR_MODEL_SAVEFILE
        },
        {   
            'model_name': KNeighborsClassifier.__name__,
            'model': KNeighborsClassifier(),
            'parameters': dict(n_neighbors = range(4, 9), # 9 is exclusive
                weights = ['uniform', 'distance'],
                algorithm=['ball_tree', 'kd_tree']),
            'filename': '../' + CONFIG.OUTPUT_DIRECTORY_NAME + CONFIG.KNN_MODEL_SAVEFILE
        }
    ]

    classifiers = {}
    table = PrettyTable()
    table.field_names = ['Model Name', 'Train Accuracy']
    
    X_train, y_train = train_df['sent_vec'].tolist(), train_df['y']
#     print(X_train.shape, y_train.shape)
    with tqdm(total=3) as bar:
        for clfDetail in classifierList:
            bar.set_description(f"Tuning {clfDetail['model_name']}")
            clf = GridSearchCV(estimator=clfDetail['model'], param_grid=clfDetail['parameters'])
            clf.fit(X_train, y_train)
            
            classifiers[clfDetail['model_name']] = {
                'model': clf,
                'best_estimators': clf.best_estimator_,
                'filename': clfDetail['filename'],
                'train_accuracy': clf.score(X_train, y_train)
            }

            table.add_row([clfDetail['model_name'], clf.score(X_train, y_train)])
            # print(f"\nModel: {clfDetail['model_name']}, Train Accuracy: {clf.score(X_train, y_train)}")
            bar.update(1)

    # Save the classifiers
    for clf_name in classifiers.keys():
        pickle.dump(classifiers[clf_name]['model'], open(classifiers[clf_name]['filename'], 'wb'))
    
    print(table)

    return classifiers

def test(classifiers: dict, test_df):
    test_results = {}
    X_test, y_test = test_df['sent_vec'].tolist(), test_df['y']
    for clf_name in classifiers.keys():
        clf = pickle.load(open(classifiers[clf_name]['filename'], 'rb'))
        test_accuracy = clf.score(X_test, y_test)

        test_results[clf_name] = {
            'test_accuracy': test_accuracy
        }

        print(f"\nModel: {clf_name}, Test Accuracy: {test_accuracy}")
    
    return test_results

def cross_val(classifiers: dict, train_df, test_df):
    '''Performs Cross Validation using the classifiers passed to this function
    and prints the performance of each classifier in terms of - 
    mean Accuracy and Standard Error
    '''
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=40)
    clf_scores = {}
    X_train, y_train = train_df['sent_vec'].tolist(), train_df['label']
    for clf_name in classifiers:
        # for repeat in tqdm(range(1,16)):
            model = classifiers[clf_name]['best_estimators']
            # scores = _evaluate_model(model, X_train, y_train)
            # scores = cross_val_score(model, X_train, y_train, scoring='f1_macro', cv=cv)
            scores = cross_validate(model, X_train, y_train, 
                                    scoring=['precision_macro', 'recall_macro', 'accuracy', 'f1_macro'], 
                                    cv=5, return_train_score=True)
            clf_scores[clf_name] = scores
#             print(f"F1 {clf_name} - {np.mean(scores):0.3f} ({sem(scores)})")
    return clf_scores

def _evaluate_model(model, X, y, repeats):
    '''A Private function called to evaluate the passed model with
    RepeatedStratified K-Fold CV using the repeats passed as argument.

    Returns: scores
    '''
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=40)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    return scores