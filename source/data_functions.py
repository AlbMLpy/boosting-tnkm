from typing import Optional

import pandas as pd
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_prepare_data(
    data_path: str, 
    test_size: float, 
    split_seed: int, 
    n_sample: Optional[int] = None,
) -> tuple[pd.DataFrame]:
    df = pd.read_csv(data_path, header=0)
    df = df.sample(n_sample, random_state=split_seed) if n_sample else df
    target_col = 'target' if 'target' in df.columns else 'class'
    return train_test_split(
        df[[v for v in df.columns if v != target_col]], 
        df[target_col], 
        test_size=test_size,
        random_state=split_seed,
    )


def scale_data(
    x, 
    x_test, 
    y, 
    y_test, 
    transform_x: bool = True, 
    transform_y: bool = True, 
    scaler: str = 'std'
) -> tuple:
    if transform_x: 
        if scaler == 'std':
            mms = StandardScaler()
        elif scaler == 'minmax':
            mms = MinMaxScaler()
        else: 
            raise ValueError(f'Bad scaler: {scaler}')
        mms.fit(x)
        x, x_test = mms.transform(x), mms.transform(x_test)  
    if transform_y:
        y_mean, y_std = y.mean(), y.std()
        y, y_test = (y - y_mean) / y_std, (y_test - y_mean) / y_std
    return x, x_test, y, y_test


def load_transform_data(
    data_path, 
    test_size, 
    split_seed, 
    transform_x: bool = True, 
    transform_y: bool = True, 
    scaler: str = 'std'
) -> tuple:
    x, x_test, y, y_test = load_prepare_data(data_path, test_size, split_seed)
    x, x_test, y, y_test = scale_data(
        x, x_test, y, y_test, transform_x, transform_y, scaler
    )
    return map(jnp.array, [x, x_test, y, y_test])


def get_df_stats(data_path) -> tuple[int, int]:
    _x1, _x2, _, _ = load_prepare_data(data_path, 0.2, split_seed=None)
    n_samples, d_dim = _x1.shape[0] + _x2.shape[0], _x1.shape[-1]
    return n_samples, d_dim


def load_transform_data_valid_test(
    data_path, 
    valid_size,
    test_size, 
    split_seed, 
    transform_x: bool = True, 
    transform_y: bool = True, 
    scaler: str = 'std',
) -> tuple[tuple, tuple]:
    x_train, x_test, y_train, y_test = load_transform_data(
        data_path, test_size, split_seed, 
        transform_x=False, transform_y=False, scaler=None,
    )
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=valid_size, 
        random_state=split_seed,
    )
    valid_split = scale_data(
        x_tr, x_val, y_tr, y_val, 
        transform_x=transform_x, 
        transform_y=transform_y, 
        scaler=scaler,
    )
    test_split = scale_data(
        x_train, x_test, y_train, y_test, 
        transform_x=transform_x, 
        transform_y=transform_y, 
        scaler=scaler,
    )
    return valid_split, test_split
