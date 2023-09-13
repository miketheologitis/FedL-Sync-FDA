# FedL-Sync-FDA

## Overview

This project is a well-structured Python implementation of the Jupyter notebook originally found at `/notebooks/completed/10_tf_sim_fda_nn.ipynb`. While the original notebook had all the code and logic in a single file, this project organizes the code into a modular, package-based structure, making it more maintainable and easy to understand.

## Usage

### Create combinations
```bash
python -m utils.create_combinations --comb_id 0 --fda naive --nn LeNet-5 --b 32 64 128 --e 50 --th 0.5 1.0
```

### Local simulation
```bash
python -m local_simulator --n_gpus 2 --n_proc 12 --comb_id 0
```