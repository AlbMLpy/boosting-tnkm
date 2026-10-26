import sys
from pathlib import Path
from collections import defaultdict
sys.path.append(str(Path.cwd().parents[1]))

import pandas as pd

from source.utils import get_path
from source.visuals import create_figure
from source.data_functions import get_df_stats
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


METRIC_COLUMNS = ["mse_t", "train_time_v_cumsum"]
DATA_NAMES_ALL = [
    "energy", "concrete", "kin8nm", "naval",
    "precipitation", "house_electric", "airline"
]
MODEL_NAMES = ["cpkm", "gb_cpkm"]
EXP_DIR = Path('./artifacts/training_artifacts')
METRIC_COL2NAME = {"mse_t": "MSE", "train_time_v_cumsum": "Time"}
MODEL_LABELS = {"cpkm": "CV-CPKM", "gb_cpkm": "GB-CPKM"}
DATA_DIR = Path.cwd().parents[1] / 'data'


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation for each rank option."""
    aggregated = (
        metrics_df
        .groupby("n_options")[METRIC_COLUMNS]
        .agg(["mean", "std"])
    )
    aggregated.columns = [
        f'{"m" if statistic == "mean" else "s"}_{metric}'
        for metric, statistic in aggregated.columns
    ]
    return aggregated.reset_index()

def mean_std_str(mean, std, mean_dec=2, std_dec=2):
    return f"{mean:.{mean_dec}f}±{std:.{std_dec}f}"


if __name__ == "__main__":
    ## Large-Scale Problems: Results
    model2res = dict()
    for model_name in MODEL_NAMES:
        dd = defaultdict(list)
        for data_name in DATA_NAMES_ALL:
            result_dir = get_path(EXP_DIR, data_name, model_name)
            metrics_df = pd.read_csv(result_dir / "metrics.csv")
            memory_df = pd.read_csv(result_dir / "memory.csv")
            result = aggregate_metrics(metrics_df)
            result = result.merge(memory_df, on="n_options")

            max_n = result['n_options'].max()
            res_dict = result[result['n_options'] == max_n].iloc[0]
            _data_name = (
                'HouseElectric' 
                if 'house' in data_name 
                else data_name.capitalize()
            )
            dd['Data'].append(_data_name)
            for metric in METRIC_COLUMNS:
                _mean = res_dict[f'm_{metric}']
                _std = res_dict[f's_{metric}']
                mean_dec = 4 if 'mse' in metric else 2
                dd[METRIC_COL2NAME[metric]].append(
                    mean_std_str(_mean, _std, mean_dec=mean_dec, std_dec=2)
                )
            dd['Memory'].append(f"{res_dict['mem']:.0f}")
        model2res[MODEL_LABELS[model_name]] = pd.DataFrame(dd)

    stats_dict = defaultdict(list)
    for data_name in DATA_NAMES_ALL:
        n_samples, d_dim = get_df_stats(DATA_DIR / f'{data_name}.csv')
        _data_name = (
            'HouseElectric' 
            if 'house' in data_name 
            else data_name.capitalize()
        )
        stats_dict['Data'].append(_data_name)
        stats_dict['N'].append(n_samples)
        stats_dict['D'].append(d_dim)


    keys = list(model2res.keys())
    combined = pd.concat(
        [model2res[k].set_index('Data') for k in keys],
        axis=1,
        keys=keys
    )

    combined = combined.swaplevel(axis=1).sort_index(axis=1)

    final_df = combined[['MSE', 'Time', 'Memory']].copy()
    final_df.loc[:, 'N'] = stats_dict['N']
    final_df.loc[:, 'D'] = stats_dict['D']
    final_df = final_df[['N', 'D', 'MSE', 'Time', 'Memory']].reset_index()
    print(final_df)

    caption = (
        r"Performance comparison of GB-CPKM and CV-CPKM across regression benchmarks. "
        + r"We report test mean squared error (MSE), training time (seconds), and peak memory consumption (MiB). "
        + r"Each experiment is repeated over 10 runs, with results reported as mean $\pm$ one standard deviation. "
        + r"$N$ denotes the number of samples, and $D$ denotes the input dimensionality. "
        + r"Overall, GB-CPKM trains up to $6\times$ faster and uses up to $2\times$ less peak memory, on average, "
        + r"than CV-CPKM, while achieving comparable test MSE."
    )
    final_df = final_df.rename({n: fr"{n} $\downarrow$"for n in ["MSE", "Time", "Memory"]}, axis=1)
    print(
        (
            final_df
            .to_latex(
                float_format="%.2f", 
                column_format='||l||c|c||c|c||c|c||c|c||',
                caption=caption,
                label="table:hypers_search",
                multicolumn=True, 
                multicolumn_format='c||',
                multirow=True,
                index=False,
            ).replace('_', '-')
        )
    )

    ## Small-Scale Problems: Dynamical Behavior
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

