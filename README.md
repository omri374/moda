# Moda
## Models and evaluation framework for trending topics detection and anomaly detection.



Moda provides an interface for evaluating models on either univariate or multi-category time-series datasets. It further allows the user to add additional models using a scikit-learn style API. All models provided in Moda were adapted to a multi-category scenario using by wrapping a univariate model to run on multiple categories. It further allows the evaluation of models using either a train/test split or a time-series cross validation.

## Usage

### Turning an items dataset into a moda dataset:
moda uses a MultiIndex to hold the datestamp and category. All models have been adapted to accept such structure.
```
import pandas as pd
from moda.dataprep import raw_to_ts, ts_to_range

DATAPATH = "SF-311_simplified.csv"
# The dataset can be downloaded from here: https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6/data
TIME_RANGE = "3H" # Aggregate all events in the raw data into 3 hour intervals

# Read raw file
raw = pd.read_csv(DATAPATH)

# Turn the raw data into a time series (with date as a pandas DatetimeIndex)
ts = raw_to_ts(raw)

# Aggregate items per time and category, given a time interval
ranged_ts = ts_to_range(ts,time_range=TIME_RANGE)
```

### Run a model

Run one model, and extract metrics using a manually labeled set

```
from moda.evaluators import get_metrics_for_all_categories, get_final_metrics
from moda.dataprep import read_data
from moda.models import STLTrendinessDetector

model = STLTrendinessDetector(freq='24H', 
                              min_value=10,
                              anomaly_type='residual',
                              num_of_std=3, lo_delta=0)

# Take the entire time series and evaluate anomalies on all of it or just the last window(s)
prediction = model.predict(dataset)
raw_metrics = get_metrics_for_all_categories(dataset[['value']], prediction[['prediction']], dataset[['label']],
                                             window_size_for_metrics=1)
metrics = get_final_metrics(raw_metrics)

## Plot results for each category
model.plot(labels=dataset['label'])
```



### Model evaluation

Example for a train/test split and evaluation
```
model = STLTrendinessDetector(is_multicategory=True, freq='3H', min_value=10,
                              anomaly_type='and', num_of_std=3)
result = eval_models(X, y, [model], label_col_name='label', train_percent=20,
                     window_size_for_metrics=2)
```

Example for Time-series-cross-validation (using scikit-learn TimeSeriesSplit)
```
model = MovingAverageSeasonalTrendinessDetector(is_multicategory=True, freq='3H', min_value=10,            
                                                anomaly_type='and', num_of_std=3)                    
result = eval_models(X, y, [model], label_col_name='label', train_percent=20, 
                     window_size_for_metrics=2)         
```



## Models currently included:
1. Moving average based seasonality decomposition (MA adapted for trendiness detection)

A wrapper on statsmodel's seasonal_decompose. A naive decomposition which uses a moving average to remove the trend, and a convolution filter to detect seasonality. The result is a time series of residuals. In order to detect anomalies and interesting trends in the time series, we look for outliers on the decomposed trend series and the residuals series. Points are considered outliers if their value is higher than a number of standard deviations of the historical values in a previous window. We evaluated different policies for trendiness prediction: 1. residual anomaly only, 2. trend anomaly only, residual OR trend anomaly, residual AND trend anomaly. Figure 6 shows an example of such method and the means to detect anomalies.

2. Seasonality and trend decomposition using Loess (Adapted STL)

STL uses iterative Loess smoothing to obtain an estimate of the trend and then Loess smoothing again to extract a changing additive seasonal component. It can handle any type of seasonality, and the seasonality value can change over time. We used the same anomaly detection mechanism as the moving-average based seasonal decomposition.
Wrapper on (https://github.com/jrmontag/STLDecompose)

3. Azure anomaly detector

Use the Azure Anomaly Detector cognitive service as a black box for detecting anomalies. Azure Anomaly finder provides an upper bound that can be used to estimate the degree of anomaly.

4. Twitter

A wrapper on Twitter's AnomalyDetection package (https://github.com/Marcnuth/AnomalyDetection)

5. LSTMs

Trains a forecasting LSTM model, and compares the prediction value at time t vs. the actual value at time t. Then, estimate the difference by comparison to the standard deviation of previous differences.


