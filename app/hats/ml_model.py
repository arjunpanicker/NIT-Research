import fasttext
import os

from .config import *
from . import utility as ut

class Model:
    _deviceList = ['ac', 'tv', 'fan', 'light', 'geyser']
    
    @staticmethod
    def createFasttextModel(filename: str):
        path: str = os.path.join(CONFIG.PWD, CONFIG.OUTPUT_DIRECTORY_NAME)
        ft_model = fasttext.train_supervised(os.path.join(path, filename), \
                                    dim=CONFIG.FT_DIMS, lr=0.5, epoch=40, verbose=1)
        ft_model.save_model(os.path.join(path, 'ft_model.ftz'))
        return ft_model

    @staticmethod
    def modelpredict(command: str) -> str:
        # TODO: Complete this function
        pass

    @staticmethod
    def predict(command: str, ft_model) -> str:
        devicePresent: bool = ut.device_exists(command, ft_model)