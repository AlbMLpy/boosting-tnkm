import jax.numpy as jnp


N_TRIALS = 10
VAL_SIZE = 0.1
TEST_SIZE = 0.2
SCALER = 'minmax'
TRANSFORM_X = True
TRANSFORM_Y = True

SAVE_FILE_PER = 'metrics.csv'
SAVE_FILE_MEM = 'memory.csv'
DATASETS = [
    'concrete', 'energy', 'kin8nm', 'naval',
    'airline', 'house_electric', 'precipitation',
]
MODELS = ['cpkm', 'gb_cpkm', 'gb_cpkm_hybrid']
LARGE_SCALE_DATA = ['airline', 'house_electric', 'precipitation']


def data2params(data_name: str) -> tuple[int, int, int]:
    base_rank, n_estimators = 2, 10
    if data_name == 'energy':
        m_order = 4
    elif data_name == 'concrete':
        m_order = 8
    elif data_name == 'kin8nm':
        m_order = 8
    elif data_name == 'naval':
        m_order = 4
    elif data_name == 'airline':
        m_order, base_rank, n_estimators = 5, 2, 5
    elif data_name == 'house_electric':
        m_order, base_rank, n_estimators = 5, 2, 10
    elif data_name == 'precipitation':
        m_order, base_rank, n_estimators = 10, 2, 10
    return m_order, base_rank, n_estimators


def get_config(data_name: str, model_name: str) -> dict:
    m_order, base_rank, n_estimators = data2params(data_name)
    rank_list = list(range(base_rank, base_rank*n_estimators + 1, base_rank))
    model_shared = dict(seed=13, dtype=jnp.float32)
    opt_shared = dict(beta_e=1.0, d_core_stop=None, reg_mode='cores')
    if model_name == 'cpkm':
        model = dict(
            type=model_name, params=dict(rank=None) | model_shared,
        )
        optim = dict(
            type='als', params=dict(n_epoch=5, gamma_w=None) | opt_shared,
        )
    elif model_name == 'gb_cpkm':
        model = dict(
            type=model_name, params=dict(rank=base_rank) | model_shared,
        )
        optim=dict(
            type='gb_als',
            params=dict(
                n_estimators=n_estimators, n_epoch=5, gamma_w=None,
            ) | opt_shared,
        )
    elif model_name == 'gb_cpkm_hybrid':
        model = dict(
            type=model_name, params=dict(rank=base_rank) | model_shared,
        )
        optim=dict(
            type='gb_als',
            params=dict(
                n_estimators=n_estimators, n_epoch=5, gamma_w=None,
            ) | opt_shared,
            optim=dict(
                type='als', params=dict(n_epoch=1, gamma_w=None) | opt_shared,
            )
        )
    exp = dict(rank_list=rank_list, gamma_list=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    features = dict(type='poly', params=dict(m_order=m_order))
    return dict(features=features, model=model, optim=optim, exp=exp)
