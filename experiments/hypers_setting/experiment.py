import sys
import argparse
import subprocess
from pathlib import Path
from itertools import product
sys.path.append(str(Path.cwd().parents[1]))

from configs.hypers_setting import DATASETS, MODELS


MODES = ("train", "memory")


def get_args():
    parser = argparse.ArgumentParser(description="Hypers Experiment")
    parser.add_argument(
        "model",
        choices=("all", *MODELS),
        help="Model to run.",
    )
    parser.add_argument(
        "data",
        choices=("all", *DATASETS),
        help="Dataset to run.",
    )
    return parser.parse_args()


def get_options(model_name: str, data_name: str) -> list[tuple[str, str]]:
    models = MODELS if model_name == "all" else [model_name]
    datasets = DATASETS if data_name == "all" else [data_name]
    return list(product(models, datasets))
    

def run_experiment(model_name: str, data_name: str) -> None:
    for mode in MODES:
        print(f"Running: {mode} | {model_name} | {data_name}")
        run_str = f"python {mode}_part.py {model_name} {data_name} -tqdm"
        subprocess.call(run_str, shell=True)


def main() -> None:
    args = get_args()
    for model_name, data_name in get_options(args.model, args.data):
        run_experiment(model_name, data_name)


if __name__ == "__main__":
    main()
