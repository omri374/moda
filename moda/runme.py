import json
import os

from moda.dataprep import read_data
from moda.evaluators import (
    get_metrics_for_all_categories,
    get_final_metrics,
    evaluate_all_models,
)
from moda.models import (
    TwitterAnomalyTrendinessDetector,
    MovingAverageSeasonalTrendinessDetector,
    STLTrendinessDetector,
    AzureAnomalyTrendinessDetector,
    LSTMTrendinessDetector,
)


def run_model(
    datapath,
    freq,
    min_date="01-01-2018",
    plot=True,
    model_name="stl",
    min_value=9,
    min_samples_for_category=100,
):
    print(
        "Loading file {0}, with frequency {1}. Model name = {2}".format(
            datapath, freq, model_name
        )
    )
    dataset = read_data(datapath, min_date=min_date)
    dataset = dataset.rename(columns={"is_anomaly": "label"})

    is_multicategory = ("category" in dataset) or ("category" in dataset.index)

    if model_name == "twitter":
        model = TwitterAnomalyTrendinessDetector(
            is_multicategory=is_multicategory,
            freq=freq,
            min_value=min_value,
            threshold=None,
            max_anoms=0.49,
            seasonality_freq=7,
        )

    if model_name == "ma_seasonal":
        model = MovingAverageSeasonalTrendinessDetector(
            is_multicategory=is_multicategory,
            freq=freq,
            min_value=min_value,
            anomaly_type="or",
            num_of_std=3,
        )

    if model_name == "stl":
        model = STLTrendinessDetector(
            is_multicategory=is_multicategory,
            freq=freq,
            min_value=min_value,
            anomaly_type="or",
            num_of_std=4,
            lo_frac=0.5,
        )

    if model_name == "azure":
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "config/config.json")
        subscription_key = get_azure_subscription_key(filename)
        model = AzureAnomalyTrendinessDetector(
            is_multicategory=is_multicategory,
            freq=freq,
            min_value=min_value,
            subscription_key=subscription_key,
        )

    if model_name == "lstm":
        model = LSTMTrendinessDetector(freq=freq, is_multicategory=is_multicategory)

    prediction = model.predict(dataset, verbose=True)
    raw_metrics = get_metrics_for_all_categories(
        dataset[["value"]],
        prediction[["prediction"]],
        dataset[["label"]],
        window_size_for_metrics=5,
    )
    metrics = get_final_metrics(raw_metrics, summarized=False)
    print(metrics)

    ## Plot each category
    if plot:
        _, file = os.path.split(datapath)
        print("Plotting...")
        model.plot(labels=dataset["label"], postfix=file)

    return prediction


def get_azure_subscription_key(file):
    try:
        with open(file) as f:
            data = json.load(f)
            return data["subscription_key"]
    except Exception as e:
        raise Exception(
            "Error loading Azure subscription key for Azure Anomaly Finder.\n"
            "Please create a json file and put your subscription_key value in it.\n"
            "See https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/apps-anomaly-detection-api\n"
            + str(e)
        )


if __name__ == "__main__":

    freqs = {"1": "30min", "2": "1H", "3": "3H", "4": "12H", "5": "24H"}
    cities = {"1": "Corona", "2": "Pompano", "3": "SF"}
    models = {"s": "stl", "m": "ma_seasonal", "t": "twitter", "a": "azure", "l": "lstm"}

    datapath = input("Enter file name or press enter for predefined datasets: ")

    freq = 0
    while freq not in ["1", "2", "3", "4", "5"]:
        freq = input(
            "Select time frequency: 30min (1), 1H (2), 3H (3), 12H (4) or 24H (5): "
        )

    if len(datapath) == 0:
        city = 0
        while city not in ["1", "2", "3", "9"]:
            city = input("Select city: Corona (1), Pompano (2), SF (3), all (9):")
        datapath = "datasets/{0}{1}_labeled.csv".format(cities[city], freqs[freq])

    inp1 = ""
    while inp1 not in ["r", "e"]:
        inp1 = input("Run one model (r) or evaluate all models? (e)?")

    if inp1 == "r":
        model = input(
            "Select model: Moving Averages (m), STL (s), Twitter (t), Azure Anomaly Detector (a), LSTM (l):"
        )
        prediction = run_model(
            datapath=datapath, freq=freqs[freq], model_name=models[model]
        )
    if inp1 == "e":
        print("Loading file {0}. Evaluating all models".format(datapath))
        evaluate_all_models(datapath=datapath, freq=freqs[freq])

if __name__ == "__main__":
    run
