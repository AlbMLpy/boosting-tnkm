import sys
import argparse
from pathlib import Path
from functools import partial
sys.path.append(str(Path.cwd().parents[1]))

import jax
jax.config.update("jax_enable_x64", True)

from source.exp_functions import train_n_trials_ranks
from source.utils import get_path, create_dir_if_not_exists
from source.data_functions import load_transform_data_valid_test
from configs.base_learners import (
    SCALER, 
    EXP_DIR,
    N_TRIALS,
    VAL_SIZE,
    DATASETS,
    TEST_SIZE,
    TRANSFORM_X, 
    TRANSFORM_Y,
    SAVE_FILE_PER,
    get_config,
)


DATA_DIR = Path.cwd().parents[1] / 'data'
TQDM_HELP = "Turn on/off tqdm interactive progress line;"


def get_args():
    parser = argparse.ArgumentParser(description="Base-Learner Rank Search")
    parser.add_argument(
        "data",
        choices=DATASETS,
        help="Dataset to run.",
    )
    parser.add_argument(
        "base_rank", type=int, default=1, help="Base Learner rank.",
    )
    parser.add_argument(
        "n_est", type=int, default=1, help="# Estimators.",
    )
    parser.add_argument(
        '-tqdm', 
        '--tqdm_enable', 
        action="store_true", 
        help=TQDM_HELP,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    data_name, base_rank, n_est = args.data, args.base_rank, args.n_est
    config = get_config(data_name, base_rank, n_est)
    res_dir = get_path(EXP_DIR, data_name, f"br{base_rank}_nest{n_est}")
    create_dir_if_not_exists(res_dir)
    save_path = str(res_dir / SAVE_FILE_PER)

    load_data_f = partial(
        load_transform_data_valid_test, 
        data_path=DATA_DIR / f'{data_name}.csv', 
        valid_size=VAL_SIZE, 
        test_size=TEST_SIZE, 
        transform_x=TRANSFORM_X, 
        transform_y=TRANSFORM_Y, 
        scaler=SCALER,
    )
    train_n_trials_ranks(
        config, 
        N_TRIALS, 
        load_data_f, 
        save_path, 
        not args.tqdm_enable,
        test_last=True,
    )
