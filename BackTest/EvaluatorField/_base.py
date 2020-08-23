'''
The base class for all the evaluator.

'''

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping

import numpy as np
import pandas as pd


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self._params = set()

    @abstractmethod
    def evaluate(self, X):
        ''' evaluate model '''

    def _check_data(self, X):
        if isinstance(X, pd.DataFrame):
            self._check_data_df(X)
        elif isinstance(X, Mapping):
            self._check_data_np(X)
        else:
            raise ValueError("The data should be dataframe or mapping !")

    def _check_data_np(self, X):
        for para in self._params:
            if para not in X.keys():
                raise ValueError("The mapping doesn't has the key %s !" % para)

    def _check_data_df(self, X):
        for para in self._params:
            if para not in X.columns:
                raise ValueError("The dataframe doesn't has the column %s !" % para)
