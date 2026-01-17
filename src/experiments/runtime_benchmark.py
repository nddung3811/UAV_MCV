from __future__ import annotations
import time
from typing import List

# ĐÚNG: import từ single_run.py
from .single_run import run_experiment


def run_single_trial(seed: int) -> float:
    """
    Run one GA–OBL trial with a given random seed
    and return execution time (seconds).
    """
    t_start = time.perf_counter()

    run_experiment(seed=seed)

    t_end = time.perf_counter()
    return t_end - t_start


def benchmark_runtime(num_runs: int = 20) -> List[float]:
    """
    Execute multiple trials and collect runtime statistics.
    """
    times: List[float] = []

    for seed in range(num_runs):
        elapsed = run_single_trial(seed)
        times.append(elapsed)
        print(f"Run {seed:02d}: {elapsed:.3f} s")

    return times


def main():
    """
    Entry point for GA–OBL runtime benchmarking.
    """
    num_runs = 20
    times = benchmark_runtime(num_runs)

    print("\nGA–OBL Runtime Benchmark")
    print(f"  Runs     : {num_runs}")
    print(f"  Avg time : {sum(times)/len(times):.3f} s")
    print(f"  Min time : {min(times):.3f} s")
    print(f"  Max time : {max(times):.3f} s")


if __name__ == "__main__":
    main()
