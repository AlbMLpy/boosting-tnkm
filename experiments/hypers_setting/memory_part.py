import sys
import warnings
import argparse
from pathlib import Path
from functools import partial
from collections import defaultdict
sys.path.append(str(Path.cwd().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import jax
import pandas as pd
from tqdm import tqdm
from memory_profiler import memory_usage
jax.config.update("jax_enable_x64", True)

from source.data_functions import load_transform_data
from source.exp_functions import train_n_trials, config2train_f
from source.utils import (
    get_path,
    update_dict, 
    update_results_dict,
    create_dir_if_not_exists,
)
from configs.hypers_setting import (
    MODELS,
    SCALER, 
    DATASETS,
    TEST_SIZE, 
    TRANSFORM_X, 
    TRANSFORM_Y,
    SAVE_FILE_MEM,
    get_config,
)


DATA_DIR = Path.cwd().parents[1] / 'data'
EXP_DIR = Path('./artifacts/training_artifacts')
TQDM_HELP = "Turn on/off tqdm interactive progress line;"


def get_args():
    parser = argparse.ArgumentParser(description='Memory Check')
    parser.add_argument(
        "model",
        choices=MODELS,
        help="Model to run.",
    )
    parser.add_argument(
        "data",
        choices=DATASETS,
        help="Dataset to run.",
    )
    parser.add_argument(
        '-tqdm', '--tqdm_enable', action="store_true", help=TQDM_HELP,
    )
    return parser.parse_args()


def training(config, load_data_f):
    train_f = config2train_f(config)
    train_n_trials(load_data_f, train_f, None, n_trials=1, tqdm_disable=True)


if __name__ == '__main__':
    args = get_args()
    data_name, model_name = args.data, args.model
    tqdm_disable = not args.tqdm_enable
    config = get_config(data_name, model_name)
    res_dir = get_path(EXP_DIR, data_name, model_name)
    create_dir_if_not_exists(res_dir)
    save_path = str(res_dir / SAVE_FILE_MEM)

    load_data_f = partial(
        load_transform_data, 
        data_path=DATA_DIR / f'{data_name}.csv', 
        test_size=TEST_SIZE, 
        transform_x=TRANSFORM_X, 
        transform_y=TRANSFORM_Y, 
        scaler=SCALER,
    )
    rank_list = config['exp']['rank_list']
    res_dict = defaultdict(list)
    for i, rank in tqdm(enumerate(rank_list, start=1), disable=tqdm_disable):
        if model_name == 'cpkm': 
            dt_changes = dict(rank=rank, gamma_w=1e-2)
        elif model_name in ['gb_cpkm', 'gb_cpkm_hybrid']: 
            dt_changes = dict(n_estimators=i, gamma_w=1e-2)
        update_dict(config, dt_changes)
        mem = memory_usage((training, (config, load_data_f), {})) # MiB
        update_results_dict(
            res_dict, n_options=i, rank=rank, mem=max(mem),
        )
    pd.DataFrame(res_dict).to_csv(save_path, index=False)
