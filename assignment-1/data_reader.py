import pandas as pd
from pathlib import Path

DATA_FOLDER = "assignment-1/data/"
INPUT_FILE = "X.csv"
TARGET_FILE = "Y.csv"


def read_data():
    data_folder = Path(DATA_FOLDER)
    input = pd.read_csv(data_folder / INPUT_FILE)
    target = pd.read_csv(data_folder / TARGET_FILE)
    return input, target