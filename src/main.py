from __future__ import annotations
from .experiments.single_run import run_experiment
from . import config



def main():
    # Có thể đổi seed ở đây
    run_experiment(seed=0)


if __name__ == "__main__":
    main()
