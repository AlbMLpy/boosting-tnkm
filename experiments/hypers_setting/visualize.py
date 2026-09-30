import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[1]))

from source.visuals import create_figure


DATA_NAMES = ["energy", "concrete", "kin8nm", "naval"]
MODEL_NAMES = ["cpkm", "gb_cpkm"]
EXP_DIR = Path('./artifacts/training_artifacts')
SAVE_PATH = str(EXP_DIR / 'rank_dynamics.png')

MODEL_COLORS = {
    "cpkm": "#01abd5",
    "gb_cpkm": "#ed3920",
}

MODEL_LABELS = {"cpkm": "CV-CPKM", "gb_cpkm": "GB-CPKM"}

PLOT_CONFIGS = [
    {
        "x_col": "n_options",
        "y_col": "m_train_time_v_cumsum",
        "std_col": "s_train_time_v_cumsum",
        "ylabel": "Train Time (sec.)",
        "xlabel": "#Rank Options",
    },
    {
        "x_col": "n_options",
        "y_col": "m_mse_t",
        "std_col": "s_mse_t",
        "ylabel": "MSE (Test)",
        "xlabel": "#Rank Options",
    },
    {
        "x_col": "n_options",
        "y_col": "mem",
        "std_col": None,
        "ylabel": "Peak Memory (MiB)",
        "xlabel": "#Rank Options",
    },
]

METRIC_COLUMNS = ["mse_t", "train_time_v_cumsum"]


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
    )
