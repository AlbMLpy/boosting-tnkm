import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[1]))

from source.visuals import create_figure, PARAMS
from visualize import (
    EXP_DIR,
    SAVE_PATH,
    DATA_NAMES, 
    MODEL_NAMES, 
    PLOT_CONFIGS,
    MODEL_COLORS,
    MODEL_LABELS,
    METRIC_COLUMNS,
)


if __name__ == "__main__":
## Different Rank Base Learners:
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
