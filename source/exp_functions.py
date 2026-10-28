import time
import copy
from functools import partial
from collections import defaultdict
from collections.abc import Callable

import jax
import numpy as np
import pandas as pd
from jax import Array
from tqdm import tqdm
import jax.numpy as jnp

from tnkm.models import CPKM
from tnkm.optim import train_als
from tnkm.core.cpkm import predict_jit
from tnkm.features import (
    PolyFeature,
    ProductFeatures, 
    PolyNormFeature,
)

from source.evaluation import mse
from source.boosting import gb_train
from source.tracker import Tracker, GBTracker
from source.utils import update_dict, extend_results_dict


def build_features(config: dict, d_dim: int) -> ProductFeatures:
    cfg = config['features']
    par = cfg['params']
    if cfg['type'] == 'poly':
        return ProductFeatures(
            (PolyFeature(par['m_order'], k_col=i) for i in range(d_dim))
        )
    elif cfg['type'] == 'poly_norm':
        return ProductFeatures(
            (PolyNormFeature(par['m_order'], k_col=i) for i in range(d_dim))
        )
    raise ValueError(f"Bad features type: {cfg['type']}")

    
def build_model(config: dict, features: ProductFeatures):
    cfg = config['model']
    par = cfg['params']
    if cfg['type'] == 'cpkm':
        return CPKM(
            features, par['rank'], seed=par['seed'], dtype=par['dtype'],
        )
    elif cfg['type'] in ['gb_cpkm', 'gb_cpkm_hybrid']: # fixed-rank base learner
        return CPKM(
            features, par['rank'], seed=par['seed'], dtype=par['dtype'],
        )
    raise ValueError(f"Bad model type: {cfg['type']}")

    
def build_optimizer(config: dict):
    cfg = config['optim']
    par = cfg['params']
    if cfg['type'] == 'als':
        return partial(
            train_als, n_epoch=par['n_epoch'], gamma_w=par['gamma_w'], 
            beta_e=par['beta_e'], d_core_stop=par['d_core_stop'],
            reg_mode=par['reg_mode'], tracker=None,
        )
    elif cfg['type'] == 'gb_als':
        return partial(
            gb_train, n_estimators=par['n_estimators'], n_epoch=par['n_epoch'],
            gamma_w=par['gamma_w'], beta_e=par['beta_e'],
            d_core_stop=par['d_core_stop'], reg_mode=par['reg_mode'],
        )
    raise ValueError(f"Bad optimizer type: {cfg['type']}")

    
def train(model, optimizer, x_train, y_train):
    if optimizer.func.__name__ == 'train_als':
        optimizer(model, x_train, y_train)
    else:
        model = optimizer(model, x_train, y_train)
    return model


def config2train_f(config: dict):
    def train_f(x: Array, y: Array) -> tuple:
        _, d_dim = x.shape
        features = build_features(config, d_dim=d_dim)
        model = build_model(config, features)
        optimizer = build_optimizer(config)
        start_time = time.perf_counter()
        model = train(model, optimizer, x, y)
        jax.block_until_ready(model.params[0])
        elapsed_time = time.perf_counter() - start_time
        train_info = dict(train_time=elapsed_time)
        return model, train_info
    return train_f


def train_n_trials(
    load_data: Callable,
    train_model: Callable,
    tracker: object | None,
    n_trials: int = 10,
    tqdm_disable: bool = False,
    logger = None,
) -> None:
    for trial in tqdm(range(1, n_trials + 1), disable=tqdm_disable):
        if logger: logger.info(f"#Trial {trial}/{n_trials} has started.")
        x_train, x_test, y_train, y_test = load_data(split_seed=trial)
        model, train_info = train_model(x_train, y_train)
        if tracker is not None:
            tracker.track(
                trial, (x_train, y_train, x_test, y_test), model, train_info,
            )


def _get_trial_path(path: str, trial: int) -> str:
    path_split = path.split('.')
    return path_split[0] + f'_trial_{trial}' + f'.{path_split[1]}'


def get_valid_info(config: dict, valid_split: tuple) -> dict:
    x_tr, x_val, y_tr, y_val = valid_split
    if config['model']['type'] == 'cpkm':
        tracker = Tracker(x_val, y_val)
        for gamma_w in config['exp']['gamma_list']:
            for rank in config['exp']['rank_list']:
                cfg = copy.deepcopy(config)
                update_dict(cfg, dict(rank=rank, gamma_w=gamma_w))
                train_f = config2train_f(cfg)
                model, train_info = train_f(x_tr, y_tr)
                tracker.track(rank, gamma_w, model, train_info)
        _res_dict = tracker.res_dict
    elif config['model']['type'] in ['gb_cpkm', 'gb_cpkm_hybrid']:
        _res_dict = defaultdict(list)
        for gamma_w in config['exp']['gamma_list']:
            cfg = copy.deepcopy(config)
            update_dict(cfg, dict(gamma_w=gamma_w))
            features = build_features(cfg, d_dim=x_tr.shape[1])
            model = build_model(cfg, features)
            optimizer = build_optimizer(cfg)
            pred_f = partial(predict_jit, fmap=features)
            tracker = GBTracker(x_tr, y_tr, x_val, y_val, pred_f, gamma_w)
            model = optimizer(model, x_tr, y_tr, tracker=tracker)
            extend_results_dict(_res_dict, **tracker.res_dict)
    else: 
        raise ValueError(f"Bad model type: {config['model']['type']}")
    df_metrics = pd.DataFrame(_res_dict)
    gamma2metric = df_metrics.groupby('gamma_w')['mse_v'].mean()
    gamma_opt = float(gamma2metric.index[gamma2metric.argmin()])
    df_metrics = df_metrics[df_metrics['gamma_w'] == gamma_opt].reset_index(drop=True)
    return df_metrics


def get_test_info(
    config: dict, 
    valid_info: pd.DataFrame, 
    test_split: tuple,
    test_last: bool,
) -> dict:
    x_train, x_test, y_train, y_test, = test_split
    test_info = defaultdict(list)
    gamma_w = valid_info['gamma_w'].unique().item()
    range_i = range(1, len(config['exp']['rank_list']) + 1)
    for i in range_i:
        cfg = copy.deepcopy(config)
        top_idx = jnp.argmin(jnp.asarray(valid_info['mse_v'][:i])).item()
        if cfg['model']['type'] == 'cpkm':
            rank_t = int(valid_info['rank_v'][top_idx])
            update_dict(cfg, dict(rank=rank_t, gamma_w=gamma_w))
        elif cfg['model']['type'] in ['gb_cpkm', 'gb_cpkm_hybrid']:
            n_estimators = int(valid_info['est_n'][top_idx])
            rank_t = n_estimators*cfg['model']['params']['rank']
            update_dict(cfg, dict(n_estimators=n_estimators, gamma_w=gamma_w))
        else:
            raise ValueError(f"Bad model type: {cfg['model']['type']}")
        if not test_last or i == len(config['exp']['rank_list']):
            train_f = config2train_f(cfg)
            model, _ = train_f(x_train, y_train)

            if cfg['optim'].get('optim') is not None: # Hybrid setting!
                cfg['optim']['optim']['params']['gamma_w'] = gamma_w
                optimizer = build_optimizer(cfg['optim'])
                model = train(model, optimizer, x_train, y_train)

            mse_value = mse(y_test, model.predict(x_test)).item()
        else:
            mse_value = np.float64(np.nan)
        test_info['n_options'].append(i)
        test_info['rank_t'].append(rank_t)
        test_info['mse_t'].append(mse_value)
    return test_info


def train_trial(
    trial: int,
    config: dict,
    load_data_f: Callable,
    save_path: str | None,
    test_last: bool = False,
):
    valid_split, test_split = load_data_f(split_seed=trial)
    valid_info = get_valid_info(config, valid_split)
    test_info = get_test_info(config, valid_info, test_split, test_last)
    df_metrics = pd.concat(
        [pd.DataFrame(valid_info), pd.DataFrame(test_info)], 
        axis=1,
    )
    df_metrics['train_time_v_cumsum'] = df_metrics['train_time_v'].cumsum()
    if save_path is not None:
        df_metrics.to_csv(_get_trial_path(save_path, trial))


def train_n_trials_ranks(
    config: dict,
    n_trials: int,
    load_data_f: Callable,
    save_path: str,
    tqdm_disable: bool = False,
    parallel: bool = False,
    test_last: bool = False,
):
    if parallel:
        raise NotImplementedError()
    else:
        for trial in tqdm(range(n_trials + 1), disable=tqdm_disable):
            _save_path = None if trial == 0 else save_path # compile - trial 0
            train_trial(trial, config, load_data_f, _save_path, test_last)
    # Combine all trials:
    df_list = []
    for trial in range(1, n_trials + 1):
        df_metrics = pd.read_csv(
            _get_trial_path(save_path, trial), index_col=0,
        )
        df_metrics['trial'] = trial
        df_list.append(df_metrics)
    pd.concat(df_list, axis=0).to_csv(save_path)
