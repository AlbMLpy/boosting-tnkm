from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
sns.set_theme()

from source.utils import get_path


PARAMS = {
    'figure.figsize': (17, 7),
    'figure.constrained_layout.use': True,
    'figure.facecolor': 'white',
    'font.size': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 16,
    'figure.max_open_warning': 50,
}


def _aggregate_metrics(
    metrics_df: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Calculate mean and standard deviation for each rank option."""
    aggregated = (
        metrics_df
        .groupby("n_options")[metric_columns]
        .agg(["mean", "std"])
    )
    aggregated.columns = [
        f'{"m" if statistic == "mean" else "s"}_{metric}'
        for metric, statistic in aggregated.columns
    ]
    return aggregated.reset_index()


def _load_model_metrics(
    experiment_dir: Path,
    data_name: str,
    model_name: str,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Load and combine metrics and memory measurements."""
    result_dir = get_path(experiment_dir, data_name, model_name)
    metrics_df = pd.read_csv(result_dir / "metrics.csv")
    memory_df = pd.read_csv(result_dir / "memory.csv")
    result = _aggregate_metrics(metrics_df, metric_columns)
    result = result.merge(memory_df, on="n_options")
    return result


def _load_dataset_metrics(
    experiment_dir: Path,
    data_name: str,
    model_names: list[str],
    metric_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Load metrics for all models for one dataset."""
    return {
        model_name: _load_model_metrics(
            experiment_dir,
            data_name,
            model_name,
            metric_columns,
        )
        for model_name in model_names
    }


def plot_model(
    axis,
    data: pd.DataFrame,
    model_name: str,
    x_col: str,
    y_col: str,
    std_col: str | None,
    model_colors: dict,
    model_labels: dict,
) -> None:
    """Plot one model's mean curve and optional standard deviation."""
    x = data[x_col]
    y = data[y_col]
    color = model_colors[model_name]
    if std_col is not None:
        std = data[std_col]
        axis.fill_between(
            x,
            y - std,
            y + std,
            color=color,
            alpha=0.2,
        )
    axis.plot(
        x,
        y,
        "*-",
        color=color,
        label=model_labels[model_name],
    )


def configure_axis(
    axis,
    data_by_model: dict[str, pd.DataFrame],
    plot_config: dict,
    data_name: str,
    row_index: int,
    column_index: int,
    model_colors: dict,
    model_labels: dict,
    plot_configs: list[dict],
) -> None:
    """Configure labels, ticks, grid, and curves for one subplot."""
    x_col = plot_config["x_col"]
    y_col = plot_config["y_col"]
    std_col = plot_config["std_col"]

    for model_name, data in data_by_model.items():
        plot_model(
            axis, data, model_name, x_col, y_col, 
            std_col, model_colors, model_labels,
        )

    rank_options = sorted({
        rank
        for data in data_by_model.values()
        for rank in data[x_col]
    })

    yscale = plot_config.get("yscale", 'linear')
    axis.set_yscale(yscale)
    axis.set_xticks(rank_options)
    if plot_config.get('xlocator', None):
        axis.xaxis.set_major_locator(
            MultipleLocator(plot_config.get('xlocator', 1))
        )
    axis.set_xlabel(plot_config.get("xlabel", ""))
    axis.set_ylabel(plot_config["ylabel"] if column_index == 0 else "")
    axis.grid(True)
    if row_index < len(plot_configs) - 1:
        axis.set_xlabel("")

    if row_index == 0:
        axis.set_title(data_name.capitalize())


def add_model_legend(fig, axes, model_names: list[str]) -> None:
    """Add one figure-level legend using handles from all axes."""
    handles, labels = [], []
    for axis in axes.flat:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)

    unique = dict(zip(labels, handles))

    fig.legend(
        unique.values(),
        unique.keys(),
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.95),
    )


def create_figure(
    experiment_dir: Path,
    data_names: list[str],
    model_names: list[str],
    metric_columns: list[str],
    plot_configs: list[dict],
    model_colors: dict,
    model_labels: dict,
    save_path=None,
    show_plot=True,
    dpi=1000,
    rc_params: dict = PARAMS,
):
    """Create the complete dynamics figure."""
    with mpl.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=len(plot_configs),
            ncols=len(data_names),
            squeeze=False,
        )
        for column_index, data_name in enumerate(data_names):
            data_by_model = _load_dataset_metrics(
                experiment_dir,
                data_name,
                model_names,
                metric_columns,
            )
            for row_index, plot_config in enumerate(plot_configs):
                configure_axis(
                    axes[row_index, column_index],
                    data_by_model,
                    plot_config,
                    data_name,
                    row_index,
                    column_index,
                    model_colors,
                    model_labels,
                    plot_configs,
                )
        add_model_legend(fig, axes, model_names)
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi)
        if show_plot:
            plt.show()
    return fig, axes
