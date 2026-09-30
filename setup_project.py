import os
import zipfile
import warnings
import urllib.request
from pathlib import Path

import gdown
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


DATA_DIR = Path('data')
DATA2ID = {'energy': 242, 'concrete': 165}
UCI_PART = 'https://archive.ics.uci.edu/static/public/'


def load_data():
    load_kin8nm()
    load_naval()
    load_house_electric()
    load_precipitation()
    load_airline()
    for data_name in DATA2ID.keys():
        load_uci(data_name)


def load_uci(data_name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from ucimlrepo import fetch_ucirepo  
    dataset = fetch_ucirepo(id=DATA2ID[data_name]) 
    x, y = dataset.data.features, dataset.data.targets 
    yv = y.values[:, 0] if y.ndim > 1 else y.values
    x = x.assign(target=yv)
    x.to_csv(DATA_DIR / f'{data_name}.csv', index=None)
    print(f'{data_name.capitalize()} Dataset downloaded ✅')


def _load_url(url: str, name_data: str) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = DATA_DIR / 'temp.zip'
    urllib.request.urlretrieve(url, filename=temp_file)
    print(f'{name_data.capitalize()} Dataset downloaded ✅')
    return temp_file


def load_house_electric() -> None:
    data_name = "house_electric"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from ucimlrepo import fetch_ucirepo 
    dataset = fetch_ucirepo(id=235)
    df = dataset.data.features
    cols = [
        "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity", "Sub_metering_1",
        "Sub_metering_2", "Sub_metering_3",
    ]
    df = df[cols].dropna().astype(float)
    df = df.rename({"Global_active_power": "target"}, axis=1)
    df.to_csv(DATA_DIR / f'{data_name}.csv', index=None)
    print(f'{data_name.capitalize()} Dataset downloaded ✅')


def load_airline() -> None:
    def get_airline_df(path) -> pd.DataFrame:
        arr_time_f = lambda x: (
            60*np.floor(x.ArrTime/100) + np.mod(x.ArrTime, 100)
        )
        dep_time_f = lambda x: (
            60*np.floor(x.DepTime/100) + np.mod(x.DepTime, 100)
        )
        return (
            pd.read_csv(path, header=0, index_col=0)
            .drop('Year', axis=1)
            .assign(ArrTime=arr_time_f)
            .assign(DepTime=dep_time_f)
            .rename({'ArrDelay': 'target'}, axis=1)
            .reset_index(drop=True)
            .astype(int)
        )

    url = (
        "https://drive.google.com/file/d/"
        + "1PS-IMDuRtZNC35DaMnDx9Ik7df0Vjn4W/view?usp=sharing"
    )
    gdown.download(url, "dataset.zip", quiet=True)
    with zipfile.ZipFile("dataset.zip") as z:
        z.extractall(str(DATA_DIR))
    (
        get_airline_df(DATA_DIR / 'airline_raw.csv')
        .to_csv(DATA_DIR / 'airline.csv', header=True, index=False)
    )
    Path("dataset.zip").unlink(missing_ok=True)
    print(f'Airline Dataset downloaded ✅')


def load_precipitation() -> None:
    url = (
        "https://raw.githubusercontent.com/kd383/GPML_SLD/refs/heads/master"
        + "/demo/precipitation/data/precipitation3240/processed-data-2010.csv"
    )
    temp_file = _load_url(url, 'Precipitation')
    data_path = DATA_DIR / 'precipitation.csv'
    os.rename(temp_file, data_path)
    cols = ['latitude', 'longitude', 'time', 'precipitation']
    (
        pd.read_csv(data_path)[cols]
        .rename({'precipitation': 'target'}, axis=1)
        .to_csv(data_path, index=None)
    )


def load_kin8nm() -> None:
    url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv"
    temp_file = _load_url(url, 'Kin8nm')
    data_path = DATA_DIR / 'kin8nm.csv'
    os.rename(temp_file, data_path)
    (
        pd.read_csv(data_path)
        .rename({'y': 'target'}, axis=1)
        .to_csv(data_path, index=None)
    )


def load_protein() -> None:
    url = (
        "https://archive.ics.uci.edu/ml/"
        + "machine-learning-databases/00265/CASP.csv"
    )
    temp_file = _load_url(url, 'protein')
    data_path = DATA_DIR / 'protein.csv'
    os.rename(temp_file, data_path)
    (
        pd.read_csv(data_path)
        .rename({'RMSD': 'target'}, axis=1)
        .to_csv(data_path, index=None)
    )


def load_naval() -> None:
    url = (
        "https://archive.ics.uci.edu/ml/" 
        + "machine-learning-databases/00316/UCI%20CBM%20Dataset.zip"
    )
    temp_file = _load_url(url, 'naval')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        with zip_ref.open("UCI CBM Dataset/data.txt") as f:
            lines = [
                [
                    float(v) 
                    for v in line.decode("utf-8").rstrip('\n').split(' ')
                    if v != ''
                ] 
                for line in f
            ]
    df = pd.DataFrame(
        lines, 
        columns=[*[f'f{i}' for i in range(1, 17)], 'target', 'target_2']
    )
    df.drop('target_2', axis=1).to_csv(DATA_DIR / 'naval.csv', index=None)
    temp_file.unlink(missing_ok=True)


if __name__ == '__main__':
    load_data()
