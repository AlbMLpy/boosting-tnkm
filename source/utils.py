import os
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp


def query_df(df_res: pd.DataFrame, query: dict) -> pd.DataFrame:
    mask = 1
    for col, val in query.items():
       mask &= (df_res[col] == val)
    return df_res[mask]


def extend_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].extend(value.copy())


def update_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].append(value)


def create_dir_if_not_exists(directory: str) -> os.PathLike:
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_for_dump(dt: dict) -> dict:
    for k in dt.keys():
        if isinstance(dt[k][0], np.ndarray):
            dt[k] = [list(v) for v in dt[k]]
    return dt


def check_nan(w_ten):
    if jnp.any(jnp.isnan(w_ten)) or (w_ten.max().item() > 1e30):
        raise ValueError("NaNs / big numbers detected in model parameters! Stop!")

    
def update_key_dict(dt: dict, target_key, new_value) -> bool:
    """ Keys should be unique! """
    for key, value in dt.items():
        if key == target_key:
            dt[key] = new_value
            return True

        if isinstance(value, dict):
            found = update_key_dict(value, target_key, new_value)
            if found:
                return True
    return False


def update_dict(dt: dict, dt_changes: dict) -> None:
    """ Update values of dt based of values in dt_changes """
    for k, v in dt_changes.items():
        res_bool = update_key_dict(dt, k, v)
        if not res_bool: print(f'Bad key: {k}')


def get_path(path, *names):
    return Path(path / '/'.join(names))
