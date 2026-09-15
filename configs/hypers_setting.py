import jax.numpy as jnp

N_TRIALS = 10
VAL_SIZE = 0.1
TEST_SIZE = 0.2
TRANSFORM_X = True
TRANSFORM_Y = True
SCALER = 'minmax'

DATASETS = [
    'airline', 'boston', 'concrete', 'energy', 'kin8nm', 
    'naval', 'power', 'protein', 'wine_red', 'yacht'
]

def data2params(data_name: str):
    if data_name == 'yacht': 
        m_order = 2
    elif data_name == 'energy':
        m_order = 4
    elif data_name == 'concrete':
        m_order = 8
    elif data_name == 'wine_red':
        m_order = 8
    elif data_name == 'kin8nm':
        m_order = 8
    elif data_name == 'protein':
        m_order = 8
    return m_order

def get_config(data_name: str, model_name: str) -> dict:
    base_rank, n_estimators = 2, 10
    rank_list = list(range(base_rank, base_rank*n_estimators + 1, base_rank))
    m_order = data2params(data_name)
    model_shared = dict(seed=13, dtype=jnp.float32)
    opt_shared = dict(beta_e=1.0, d_core_stop=None, reg_mode='cores')
    if model_name == 'cpkm':
        model = dict(
            type=model_name, params=dict(rank=None) | model_shared,
        )
        optim = dict(
            type='als', params=dict(n_epoch=5, gamma_w=1e-2) | opt_shared,
        )
    elif model_name == 'gb_cpkm':
        model = dict(
            type=model_name, params=dict(rank=base_rank) | model_shared,
        )
        optim=dict(
            type='gb_als',
            params=dict(
                n_estimators=n_estimators, n_epoch=5, gamma_w=1e-4
            ) | opt_shared,
        )
    exp = dict(rank_list=rank_list)
    features = dict(type='poly', params=dict(m_order=m_order))
    return dict(features=features, model=model, optim=optim, exp=exp)
