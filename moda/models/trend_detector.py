import gc
import os
from abc import ABC, abstractmethod

import pandas as pd

MIN_SAMPLES_PER_CATEGORY = 5
GENERAL_CATEGORY = 'general'


class AbstractTrendDetector(ABC):

    @abstractmethod
    def __init__(self, freq, is_multicategory=False, resample=True):

        """Abstract class for single and multi-category time series anomaly detection

        Parameters
        ----------
        is_multicategory: bool
            Specifies whether the data is held with a pandas MultiIndex, where the first index is date/time and the
            second is a category. This is used for multiple time series, one for each category

        """

        self.is_multicategory = is_multicategory
        self.freq = freq
        self.resample = resample
        self.input_data = {}

    @abstractmethod
    def fit_one_category(self, dataset, category=None, verbose=False):
        pass

    def fit(self, X, y=None, verbose=False):
        """Fits the trend detection.

        Parameters
        ----------
        dataset : pandas.DataFrame
            A pandas DataFrame with either:
            * a two-leveled multi-index, the first indexing time and the second indexing class/topic frequency
            per-window, and a single column of a numeric dtype
            * a DatetimeIndex, and a value column of a numeric dtype

        """

        # Check that X and y have correct shape

        if y is not None:
            print("Ignoring value y, this model is unsupervised")

        if self.is_multicategory:
            X = X.reset_index()
            X = X.set_index('date')

        if 'category' not in X:
            return self.fit_one_category(X, category=GENERAL_CATEGORY, verbose=verbose)

        categories = X['category'].unique()
        for category in categories:
            one_category = X.loc[X['category'] == category,]

            one_category = one_category.resample(self.freq, convention='start').asfreq().fillna(0)
            one_category['category'] = category
            one_category = one_category.resample(self.freq, convention='start').asfreq().fillna(0)
            if len(one_category) < MIN_SAMPLES_PER_CATEGORY:
                continue
            # one_category = one_category.asfreq(freq = self.freq)
            self.fit_one_category(one_category, category=category, verbose=verbose)

            # Collect garbage to avoid memory issues using Tensorflow
            gc.collect()

    @abstractmethod
    def predict_one_category(self, X, category):
        pass

    def predict(self, X, verbose=False):
        output = pd.DataFrame()

        if self.is_multicategory:
            X = X.reset_index(level='category')

        if 'category' not in X:
            self.fit_one_category(X, category=GENERAL_CATEGORY)
            return self.predict_one_category(X, category=GENERAL_CATEGORY)

        categories = X['category'].unique()

        if verbose:
            print("categories found = {}".format(categories))
        category_count = 0
        for category in categories:
            one_category = self.get_one_category_df(X, category)

            res = self.predict_one_category(one_category, category)
            res['category'] = category
            output = pd.concat([output, res],sort=True)
            # Collect garbage to avoid memory issues using Tensorflow
            gc.collect()

        if self.is_multicategory:
            output = output.reset_index().set_index(['date', 'category'])

        return output

    def get_one_category_df(self, X, category):
        one_category = X.loc[X['category'] == category,]
        if one_category.index is None:
            one_category = one_category.set_index(pd.DatetimeIndex(one_category['date'])).drop(columns='date')
        if not isinstance(one_category.index, pd.DatetimeIndex):
            raise Exception("Index should be a pd.DatetimeIndex")

        if self.resample:
            one_category = one_category[~one_category.index.duplicated(keep='first')]
            one_category = one_category.resample(self.freq, convention='start').asfreq().fillna(0)
            one_category['category'] = category
        return one_category

    def _type(self):
        return self.__class__.__name__

    def plot(self, labels=None, savefig=True, postfix="", plots_path="plots/",specific_category=None):
        import matplotlib.pyplot as plt

        if self.input_data is None:
            print("No data found")

        if len(self.input_data.keys()) == 1:
            self.plot_one_category(category=GENERAL_CATEGORY,labels=labels)
            return

        if specific_category is not None:
            if labels is not None:
                category_labels = labels.reset_index(level='category')
                category_labels = category_labels.loc[category_labels['category'] == specific_category,].drop(columns='category')
            else:
                category_labels = None
            self.plot_one_category(category=specific_category,labels=category_labels)
            return


        categories = self.input_data.keys()

        category_count = 0
        for category in categories:
            if labels is not None:
                if isinstance(labels.index, pd.MultiIndex):
                    labels = labels.reset_index().set_index('date')
                category_label = labels.loc[labels['category'] == category,]
                category_label = category_label.drop(labels="category", axis=1)
            else:
                category_label = None

            plt.clf()
            plt.figure(category_count, figsize=(20, 10))
            self.plot_one_category(category=category, labels=category_label)

            if savefig:
                strFile = os.path.join(plots_path, self._type() + "-" + str(postfix) + "-" + category + ".png")
                if os.path.isfile(strFile):
                    os.remove(strFile)
                plt.savefig(strFile)

    @abstractmethod
    def plot_one_category(self, category=None, labels=None):
        pass
