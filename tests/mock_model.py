import numpy as np
import pandas as pd

from moda.models.trend_detector import AbstractTrendDetector


class MockModel(AbstractTrendDetector):

    __name__ = 'MockModel'

    def __init__(self, prediction=1,is_multicategory = True, freq = '12H'):
        super(MockModel, self).__init__(freq, is_multicategory)
        self.prediction = prediction

    def predict_one_category(self, X, category):
        new_X = X.copy(deep=False)
        new_X['prediction'] = np.full(len(new_X), self.prediction)
        return new_X

    def fit_one_category(self, dataset, category=None, verbose=False):
        pass

