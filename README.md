# Moda
Models and evaluation framework for trending topics detection

Moda is a set of models capable of detecting trending topics. Currently supported models:
- Moving Average based Seasional Decomposition (wrapping statsmodels.seasonal_decompose)
- Seasonal and Trend decomposition using Loess (STL)
- Azure Anomaly Detector
- AnomalyDetection (Wrapper on Twitter's AnomalyDetection package)

## Usage

### Model evaluation
For evaluation, there are two main usages:
- Run model
- Evaluate models

#### Run model
This functionality allows you to run a specific model included in moda on your dataset. In case you provide ground truth data, it will evaluate the requested model on your data and return an evaluation. It will also plot the models's results into a file

To run and evaluate a specific model, either run runme.py or this code:
```
dataset = read_data(datapath)
model = MovingAverageSeasonalTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                                        anomaly_type='or', num_of_std=3)
prediction = model.predict(dataset)
raw_metrics = get_evaluation_metrics(dataset[['value']], prediction[['prediction']], dataset[['label']],
                                         window_size_for_metrics=5)
metrics = get_final_metrics(raw_metrics)
print(metrics)
model.plot(labels=dataset['label'])
```


#### Evaluate models
This functionality allows you to compare different models and to optimize model parameters. see eval_models in runme.py

### Individual model usage
You can use one of Moda's models (or model wrappers) on your data, without using Moda's evaluation functionality.



