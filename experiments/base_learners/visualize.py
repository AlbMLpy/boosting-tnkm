import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[1]))

from source.visuals import create_figure, PARAMS
from configs.base_learners import BASE_RANKS, N_ESTIMATORS, EXP_DIR


DATA_NAMES = ["energy", "concrete", "kin8nm", "naval"]
MODEL_NAMES = [
    f"br{br}_nest{nest}" for br, nest in zip(BASE_RANKS, N_ESTIMATORS)
]
SAVE_PATH = str(EXP_DIR / 'base_rank_ablations.png')

_COLORS = ["#01abd5", "#ed3920", '#268c26', '#ff7f0e', '#9467bd']
MODEL_COLORS = {
    f"br{br}_nest{nest}": c 
    for br, nest, c in zip(BASE_RANKS, N_ESTIMATORS, _COLORS)
}
MODEL_LABELS = {
    f"br{br}_nest{nest}": f"Base Rank {br}" 
    for br, nest in zip(BASE_RANKS, N_ESTIMATORS)
}

PLOT_CONFIGS = [
    {
        "x_col": 'rank',
        "y_col": "m_train_time_v_cumsum",
        "std_col": None,
        "ylabel": "Train Time (sec.)",
        "xlabel": "Full Model Rank",
        "xlocator": 2,
    },
    {
        "x_col": 'rank',
        "y_col": "m_mse_v",
        "std_col": None,
        "ylabel": "MSE (Valid)",
        "yscale": 'log',
        "xlabel": "Full Model Rank",
        "xlocator": 2,
    },
]

METRIC_COLUMNS = ["mse_v", "train_time_v_cumsum"]


if __name__ == "__main__":
    create_figure(
        EXP_DIR, 
        DATA_NAMES, 
        MODEL_NAMES, 
        METRIC_COLUMNS,
        PLOT_CONFIGS,
        MODEL_COLORS,
        MODEL_LABELS,
        SAVE_PATH, 
        show_plot=True,
        dpi=1000,
        rc_params=PARAMS | {'legend.fontsize': 12}
    )
