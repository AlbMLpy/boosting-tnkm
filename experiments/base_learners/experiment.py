import sys
import argparse
import subprocess
from pathlib import Path
from itertools import product
sys.path.append(str(Path.cwd().parents[1]))

from configs.base_learners import DATASETS, BASE_RANKS, N_ESTIMATORS


MODES = ("train", "memory")


def get_args():
    parser = argparse.ArgumentParser(description="Base Learners Experiment")
    parser.add_argument(
        "data",
        choices=("all", *DATASETS),
        help="Dataset to run.",
    )
    return parser.parse_args()


def get_options(data_name: str) -> list[tuple[str, str]]:
    specs = list(zip(BASE_RANKS, N_ESTIMATORS))
    datasets = DATASETS if data_name == "all" else [data_name]
    return list(product(specs, datasets))
    

def run_experiment(specs: tuple[int, int], data_name: str) -> None:
    rank, n_est = specs
    for mode in MODES:
        print(f"Running: {mode}|Rank={rank}|#Est={n_est}|Data - {data_name}")
        run_str = f"python {mode}_part.py {data_name} {rank} {n_est} -tqdm"
        subprocess.call(run_str, shell=True)


def main() -> None:
    args = get_args()
    for specs, data_name in get_options(args.data):
        run_experiment(specs, data_name)


if __name__ == "__main__":
    main()
