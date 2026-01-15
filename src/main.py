from __future__ import annotations
from .experiments.single_run import run_experiment
from . import config




def main():
    # Có thể đổi seed ở đây
    #run_experiment(seed=0)
    instances = [
        "data/point4.mat",
        "data/point5.mat",
    ]

    # ===== Số MCV muốn test =====
    mcv_list = [3, 4, 5, 6, 7]

    # ===== Seed =====
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]

    for inst in instances:
        config.INSTANCE = inst

        for mcv in mcv_list:
            config.N_VEHICLE = mcv

            for s in seeds:
                run_experiment(seed=s)

if __name__ == "__main__":
    main()
