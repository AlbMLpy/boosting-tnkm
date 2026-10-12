from pathlib import Path

import jax.numpy as jnp


N_TRIALS = 10
VAL_SIZE = 0.1
TEST_SIZE = 0.2
SCALER = 'minmax'
TRANSFORM_X = True
TRANSFORM_Y = True
GAMMA_LIST = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
BASE_RANKS = [1, 2, 4, 8, 12]
N_ESTIMATORS = [24, 12, 6, 3, 2]

SAVE_FILE_PER = 'metrics.csv'
SAVE_FILE_MEM = 'memory.csv'
EXP_DIR = Path('./artifacts/training_artifacts')
DATASETS = ['concrete', 'energy', 'kin8nm', 'naval']


def data2params(data_name: str):
    if data_name == 'energy':
        m_order = 4
    elif data_name == 'concrete':
        m_order = 8
    elif data_name == 'kin8nm':
        m_order = 8
    elif data_name == 'naval':
        m_order = 4
    elif data_name == 'precipitation':
        m_order = 10
    return m_order


def get_config(data_name: str, base_rank: int, n_est: int) -> dict:
    m_order = data2params(data_name)
    rank_list = list(range(base_rank, base_rank*n_est + 1, base_rank))
    model_shared = dict(seed=13, dtype=jnp.float32)
    opt_shared = dict(beta_e=1.0, d_core_stop=None, reg_mode='cores')
    model = dict(type='gb_cpkm', params=dict(rank=base_rank) | model_shared)
    optim=dict(
        type='gb_als',
        params=dict(n_estimators=n_est, n_epoch=5, gamma_w=None,) | opt_shared,
    )
    exp = dict(rank_list=rank_list, gamma_list=GAMMA_LIST)
    features = dict(type='poly', params=dict(m_order=m_order))
    return dict(features=features, model=model, optim=optim, exp=exp)
