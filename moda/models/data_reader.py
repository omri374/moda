import pandas as pd


def read_labeled_data(filepath):
    """
    Reads a labeled datasets rom a file
    :param filepath: The path of the CSV file, which contains a 'date' column
    :return: A pandas.DataFrame
    """
    df = pd.read_csv(filepath, index_col=None)
    df['date'] = pd.to_datetime(df['date'])
    return df
