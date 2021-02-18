import pandas as pd
import numpy as np
import re
import os

from .function import overload
from .config import *

class Preprocessing:
    '''Defining a Utility class with some utility functions
    '''
    stop_words: list = []
    sms_translations_data: pd.DataFrame = pd.DataFrame()


    def __init__(self, stop_words, sms_translations_data) -> None:
        self.stop_words = stop_words
        self.sms_translations_data = sms_translations_data
        pass

    def _removeStopWords(self, line: str)-> str:
        newline = [w for w in line.split() if w not in self.stop_words]
        return ' '.join(newline)

    def _smsTranslate(self, line: str)-> str:
        newline = ''
        for root_word, col in self.sms_translations_data.iteritems():
            newline = ' '.join([word if word not in col else root_word for word in line.split()])

        return newline 

    def _convertSentToVec(self, sentence: str, ftModel)-> np.ndarray:
        sent_vec = ftModel.get_sentence_vector(sentence)
        return sent_vec

    def langTranslate(self, line: str)-> str:
        # TODO: Future implementation if required
        pass
    
    @overload
    def preprocessing(self, dataset: pd.DataFrame)-> pd.DataFrame:
        # Converting to lowercase
        dataset['commands'] = dataset['commands'].apply(str.lower)
        dataset['label'] = dataset['label'].apply(str.lower)

        # Apply __label__ to the class labels and replace space(' ') with underscore('_')
        dataset['label'] = dataset['label'].apply(lambda x: '__label__' + x)
        dataset['label'] = dataset['label'].apply(lambda x: re.sub(' ', '_', x))

        # Remove stopwords
        dataset['commands'] = dataset['commands'].apply(self._removeStopWords)

        # Perform sms translation
        dataset['commands'] = dataset['commands'].apply(self._smsTranslate)

        # Perform language translation
        # dataset['commands'] = dataset['commands'].apply(self.langTranslate)
        return dataset

    @overload
    def preprocessing(self, command: str)-> str:
        command = command.lower()
        command = self._removeStopWords(command)
        command = self._smsTranslate(command)

        return command

    def strpreprocessing(self, command: str)-> str:
        command = command.lower()
        command = self._removeStopWords(command)
        command = self._smsTranslate(command)

        return command

    def saveToCsv(self, dataset: pd.DataFrame)-> None:
        dataset.to_csv(os.path.join('../', CONFIG.OUTPUT_DIRECTORY_NAME, \
                                    CONFIG.OUTPUT_DATASET_FILE), header=False, \
                        index=False, sep=' ')

    def convertCommandToVector(self, dataset: pd.DataFrame, ftModel)-> pd.DataFrame:
        dataset['sent_vec'] = dataset['commands'].apply( \
            lambda x: self._convertSentToVec(x, ftModel))

        return dataset