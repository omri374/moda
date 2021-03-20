import numpy as np
import pandas as pd

from moda.models.trend_detector import TrendDetector, MIN_SAMPLES_PER_CATEGORY
from moda.models.twitter.anomaly_detect_ts import anomaly_detect_ts


class TwitterAnomalyTrendinessDetector(TrendDetector):
    GENERAL_CATEGORY = "general"

    __name__ = "TwitterAnomalyTrendinessDetector"

    def __init__(
        self,
        freq,
        seasonality_freq=7,
        is_multicategory=False,
        min_value=None,
        max_anoms=0.1,
        direction="pos",
        alpha=0.05,
        only_last=None,
        resample=True,
        threshold=None,
        e_value=False,
        longterm=False,
        piecewise_median_period_weeks=2,
    ):
        super(TwitterAnomalyTrendinessDetector, self).__init__(
            freq, is_multicategory, resample
        )
        self.seasonality_freq = seasonality_freq
        self.min_value = min_value
        self.piecewise_median_period_weeks = piecewise_median_period_weeks
        self.longterm = longterm
        self.e_value = e_value
        self.threshold = threshold
        self.only_last = only_last
        self.alpha = alpha
        self.direction = direction
        self.max_anoms = max_anoms
        self.results = None

    def fit_one_category(self, dataset, category=None, verbose=False):
        x = dataset["value"]

        try:
            model_result = anomaly_detect_ts(
                x,
                period=self.seasonality_freq,
                max_anoms=self.max_anoms,
                direction=self.direction,
                alpha=self.alpha,
                only_last=self.only_last,
                threshold=self.threshold,
                e_value=self.e_value,
                longterm=self.longterm,
                piecewise_median_period_weeks=self.piecewise_median_period_weeks,
            )

            anomalies = model_result["anoms"].to_frame("anomalies")
            if "anomalies" in dataset:
                dataset = dataset.drop(columns=["anomalies"])

            if len(anomalies) == 0:
                dataset_with_pred = dataset
                dataset_with_pred["anomalies"] = np.nan
            else:
                dataset_with_pred = pd.merge(
                    dataset, anomalies, how="left", right_index=True, left_index=True
                )

            dataset_with_pred["prediction"] = np.where(
                np.isnan(dataset_with_pred["anomalies"]), 0, 1
            )

        except Exception as e:
            print(e)
            dataset_with_pred = dataset
            dataset_with_pred["prediction"] = 0

        if self.min_value is not None:
            dataset_with_pred["prediction"] = np.where(
                dataset_with_pred["value"] < self.min_value,
                0,
                dataset_with_pred["prediction"],
            )

        self.input_data[category] = dataset_with_pred

        return dataset_with_pred

    def predict_one_category(self, X, category):
        results = self.input_data.get(category)

        if results is None:
            test = X
        else:
            test = pd.concat([results, X], sort=True)
            test = test[~test.index.duplicated(keep="first")]

        # We have to fit the entire datasets again (train+test)
        # as we can't compute anomalies iteratively
        if len(test) > MIN_SAMPLES_PER_CATEGORY:
            test = self.fit_one_category(test, category)

        return test

    def plot_one_category(self, category=None, labels=None):
        import matplotlib.pyplot as plt

        if self.input_data is None:
            print("Empty datasets")
            return None

        if len(self.input_data) == 0:
            print("Empty datasets")
            return None

        if category not in self.input_data.keys():
            print("Empty datasets")

        def ts_subplot(plt, series, label):
            plt.plot(series, label=label, linewidth=0.5)
            plt.legend(loc="best")
            plt.xticks(rotation=90)

        plt.subplot(411)
        ts_subplot(plt, self.input_data[category]["value"], label="Original")
        plt.subplot(412)
        ts_subplot(plt, self.input_data[category]["prediction"], label="Prediction")
        if (labels is None) and ("label" in self.input_data[category]):
            labels = self.input_data[category]["label"]

        if labels is not None:
            plt.subplot(413)
            ts_subplot(plt, labels, label="Labels")

            diff = labels - self.input_data[category]["prediction"]
            plt.subplot(414)
            ts_subplot(plt, diff, label="Difference between labeled and predicted")

        if category is None:
            plt.suptitle(
                "Twitter anomaly detection results for threshold (std)="
                + str(self.threshold)
                + ", max_anoms="
                + str(self.max_anoms)
            )
        else:
            plt.suptitle(
                "Twitter anomaly detection results for "
                "category = " + category + ", "
                "threshold (std)=" + str(self.threshold) + ", "
                "max_anoms=" + str(self.max_anoms)
            )
