# MOGA-OBL for UAV–MCV Cooperative Routing

This repository implements a **Multi-Objective Genetic Algorithm with Opposition-Based Learning (MOGA-OBL)**
for a **UAV–MCV cooperative routing and charging problem**.

The code is a clean Python refactor of an original MATLAB-based implementation,
with **no change to the underlying optimization logic**.

---

## 1. Problem Overview

The problem considers:

- **UAV (Unmanned Aerial Vehicle)** performing a reconnaissance route over a set of points.
- **MCVs (Mobile Charging Vehicles)** that travel on the ground to charge the UAV at selected locations.
- The UAV route is **fixed**.
- Decision variables determine:
  - Which MCV (if any) charges the UAV on each route edge.
  - The charging location along the edge.
  - The charging duration.

### Objectives (minimization)

1. **Total operational cost**
2. **Time window violation** (if applicable)

### Constraints

- UAV battery capacity limits
- MCV travel feasibility
- Charging time and energy limits

---

## 2. Algorithm: MOGA-OBL

The solution is based on:

- Multi-Objective Genetic Algorithm (MOGA)
- Opposition-Based Learning (OBL)
- Tournament selection
- Environmental selection (SPEA2-style fitness)
- TOPSIS-based decision making

---

## 3. Project Structure

```
src/
├─ algorithms/
│  ├─ ga.py
│  ├─ obl.py
│  └─ selection.py
│
├─ core/
│  ├─ fitness.py
│  └─ initialize.py
│
├─ decision/
│  └─ topsis.py
│
├─ experiments/
│  └─ single_run.py
│
├─ io/
│  └─ mat.py
│
├─ visualization/
│  ├─ pf.py
│  └─ routes.py
│
├─ config.py
└─ main.py
```

---

## 4. Installation

### Requirements

- Python 3.9+
- numpy
- scipy
- matplotlib

### Install dependencies

```
pip install numpy scipy matplotlib
```

---

## 5. Configuration

Edit `src/config.py` to define problem parameters such as:

- Number of UAVs
- Number of MCVs
- Start location
- Input data file (.mat)

---

## 6. Running the Code

Run a single experiment from the project root:

```
python -m src.main
```

or

```
python -m src.experiments.single_run
```

---

## 7. Output

Results are saved in the `results/` directory, including:

- Best objective values
- UAV route plots
- MCV charging route plots
- Pareto front visualization

---

## 8. Reproducibility

All random processes use fixed seeds.
Running with the same seed produces identical results.

---

## 9. Algorithm Name

**MOGA-OBL**
(Multi-Objective Genetic Algorithm with Opposition-Based Learning)

---

## 10. License

For academic and research use.
