# Moda

## Models and evaluation framework for trending topics detection and anomaly detection

![CI](https://github.com/omri374/moda/workflows/CI/badge.svg?branch=master)
[![PyPI package version](https://badge.fury.io/py/moda.svg)](https://badge.fury.io/py/moda.svg)

Moda provides an interface for evaluating models on either univariate or multi-category
time-series datasets. It further allows the user to add additional models by using
a scikit-learn style API. All models provided in Moda were adapted
to a multi-category scenario by extending a univariate model
to run on multiple categories.
It further allows the evaluation of models using either
a train/test split or a time-series cross validation.

## Installation

`pip install moda`

## Usage

### Turning a raw dataset into a moda dataset

moda uses a MultiIndex to hold the datestamp and category.
All models have been adapted to accept such structure.

The input dataset is assumed to have an entry per row and a datestamp
column called 'date'. An additional 'category' column is optional.
As a first step, the dataset is aggregated to a fixed size time interval,
and a new dataset with a 'date','category' (optional) and 'value' columns is created.

A MultiIndex of 'date' (pandas DatetimeIndex) and 'category' is the dataset's index.

```python
import pandas as pd
from moda.dataprep import raw_to_ts, ts_to_range

raw_data_path = f"example/SF_data/SF311-2008.csv"
# The full dataset can be downloaded from here: 
# https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6/data
TIME_RANGE = f"24H" # Aggregate all events in the raw data into 24 hours intervals

# Read raw file
raw = pd.read_csv(raw_data_path)

# Turn the raw data into a time series (with date as a pandas DatetimeIndex)
ts = raw_to_ts(raw)

# Aggregate items per time and category, given a time interval
ranged_ts = ts_to_range(ts=ts,time_range=TIME_RANGE)
```

### Model evaluation

```python
from moda.evaluators import get_metrics_for_all_categories, get_final_metrics
from moda.dataprep import read_data
from moda.models import STLTrendinessDetector

dataset = read_data(f"datasets/SF24H_labeled.csv")
print(dataset.head())

model = STLTrendinessDetector(freq=f'24H', 
                              min_value=10,
                              anomaly_type=f'residual',
                              num_of_std=3, lo_delta=0)

# Take the entire time series and evaluate anomalies 
# on all of it or just the last window(s)
prediction = model.predict(dataset)
raw_metrics = get_metrics_for_all_categories(dataset[['value']], 
                                             prediction[['prediction']], 
                                             dataset[['label']],
                                             window_size_for_metrics=1)
metrics = get_final_metrics(raw_metrics)
print(f"f1 = {metrics['f1']}")
print(f"precision = {metrics['precision']}")
print(f"recall = {metrics['recall']}")

## Plot results for each category
#model.plot(labels=dataset['label'])   
```

## Examples

A jupyter notebook with this example can be found [here](example.ipynb).

A more detailed example which includes an exploratory data analysis
can be found [here](moda/example/EDA.ipynb)

## Models currently included

1. Moving average based seasonality decomposition (MA adapted for trendiness detection)

    A wrapper on `statsmodel`'s seasonal_decompose.
    A naive decomposition which uses a moving average to remove the trend,
    and a convolution filter to detect seasonality.
    The result is a time series of residuals.

    In order to detect anomalies and interesting trends in the time series,
    we look for outliers on the decomposed trend series and the residuals series.
    Points are considered outliers if their value is higher than a number of
    standard deviations of the historical values in a previous window.
    We evaluated different policies for trendiness prediction:

    1. Residual anomaly only
    2. Trend anomaly only
    3. Residual OR trend anomaly
    4. Residual AND trend anomaly

    This is the baseline model, which gives decent results when seasonality
    is more or less constant.

1. Seasonality and trend decomposition using Loess (Adapted STL)

    STL uses iterative Loess smoothing to obtain an estimate of the trend and then
    Loess smoothing again to extract a changing additive seasonal component.
    It can handle any type of seasonality, and the seasonality value can change over time.
    We used the same anomaly detection mechanism as the moving-average based
    seasonal decomposition.
    Wrapper on (<https://github.com/jrmontag/STLDecompose>)
    Use this model when trend and seasonality have a more complex pattern.
    It usually outperforms the moving average model.

    Example output plot for STL:
    ![STL](https://github.com/omri374/moda/raw/master/figs/STL_example.png)
    The left hand side shows the origin (top) and
    decomposed time series (Seasonal, trend, residual)
    The right hand side shows anomalies found on the residuals time series (top),
    trend, prediction (combination of residuals and trend anomalies),
    and ground truth (bottom).

1. Azure anomaly detector

    Use the Azure Anomaly Detector cognitive service as a black box for detecting anomalies.
    Azure Anomaly finder provides an upper bound that can be used to estimate
    the degree of anomaly.
    This model is useful when the anomalies have a relatively complex structure.

1. Twitter

    A wrapper on Twitter's AnomalyDetection package
    (<https://github.com/Marcnuth/AnomalyDetection>)
    This model is similar to (1) and (2), but has a more sophisticated way of
    detecting the anomalies once the time series is analyzed.

1. LSTMs

    Trains a forecasting LSTM model, and compares the prediction value at time t vs.
    the actual value at time t. Then, estimate the difference by comparison to the
    standard deviation of previous differences.
    This is useful only when there exists enough data for representing
    the time series pattern.

    An example on running LSTMs can be found [here](moda/example/lstm/LSTM_AD.ipynb).

## Running tests and linting

`Moda` uses pytest for testing. In order to run tests, just call `pytest`
from `Moda`'s main directory. For linting, this module uses PEP8 conventions.
