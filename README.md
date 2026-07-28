# Fractional Stochastic Transportation - Computational Validation

This repository contains the computational framework, simulation models, and generated results accompanying the paper **"A Fractional Stochastic Transportation Model"**. 

It provides the necessary Python source code, generated inputs, and reproducibility scripts to validate the Progressive Hedging (PH) implementation, verify the carbon-trading separation property, and assess the operational impact of fractional memory.

## Repository Structure

```
├── computational_validation/
│   ├── src/
│   │   ├── model_builder.py        # Core Mixed-Integer Linear Programming (MILP) models (PuLP/Gurobi)
│   │   ├── progressive_hedging.py  # Progressive Hedging (PWL-PH) algorithm implementation
│   │   ├── scenario_tree.py        # Tree structure and demand generation
│   │   └── experiments.py          # Large-scale experiment orchestration
│   ├── results/
│   │   └── ph_history.csv          # Raw iteration trajectory for the PH consensus variables
│   └── gurobi.log                  # Exact solver logs
├── generate_all_tables.py          # Main script to reproduce all tables from the paper
├── generate_manifests.py           # Utility to compute file checksums
├── MANIFEST.txt                    # List of included files
└── SHA256SUMS.txt                  # Cryptographic checksums ensuring data integrity
```

## Prerequisites

To run the models, you will need:
- **Python 3.8+**
- **PuLP** (`pip install pulp`)
- **Gurobi Optimizer** (A valid academic/commercial license is required for `gurobipy`)

To install the Python dependencies, you can use:
```bash
pip install -r requirements.txt # (if provided, otherwise manually install pulp)
```
*Note: The model strictly expects Gurobi to solve the subproblems to optimality within the prescribed gap ($10^{-4}$).*

## Reproducing the Results

To reproduce exactly the tables presented in the computational section of the manuscript (Tables 2, 3, 3B, 4, 5, 5B, and 6), simply run:

```bash
python generate_all_tables.py
```

This script will:
1. Initialize the minimal multistage scenario tree.
2. Formulate and solve the **Deterministic Equivalent (DE)** model to certified global optimality.
3. Formulate and run the **Piecewise-Linear Progressive Hedging (PWL-PH)** approximation.
4. Extract the continuous non-anticipative residual trajectory and recover the binary policy.
5. Solve comparative variants of the model (Fractional memory, First-order memory, and No operational memory).
6. Output all formatted tables directly into the console/file.

## Data Availability & Zenodo Integration

This repository is designed to be archived via **Zenodo**. The files `MANIFEST.txt` and `SHA256SUMS.txt` guarantee the immutability of the experimental inputs and outputs. 

To verify the integrity of the downloaded files, you can run:
```bash
shasum -a 256 -c SHA256SUMS.txt
```
