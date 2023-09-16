# FedL-Sync-FDA

## Overview

This project is a well-structured Python implementation of the Jupyter notebook originally found at `/notebooks/completed/10_tf_sim_fda_nn.ipynb`. While the original notebook had all the code and logic in a single file, this project organizes the code into a modular, package-based structure, making it more maintainable and easy to understand.

# Usage

## Combination Script

### Parameters
`--comb_id`

- **Type**: Integer
- **Description**: The prefix for the combinations file. The generated combinations will be saved to a file named `<PREFIX>.json`.
- **Required**: Yes
- **Example**: `--comb_id 1`

`--b`

- **Type**: List of integers
- **Description**: Specifies the batch size(s) for the training.
- **Required**: No
- **Example**: `--b 32 64 128` (This will generate combinations with batch sizes of 32, 64, and 128.)

`--e`

- **Type**: Integer
- **Description**: Specifies the number of epochs for training.
- **Required**: Yes
- **Example**: `--e 10`

`--fda`

- **Type**: List of strings
- **Description**: Specifies the FDA name(s). 
- **Required**: Yes
- **Example**: `--fda linear naive sketch`

`--nn`

- **Type**: List of strings
- **Description**: Specifies the name(s) of the CNN models. Currently, it supports 'LeNet-5' and 'AdvancedCNN'.
- **Required**: Yes
- **Example**: `--nn LeNet-5 AdvancedCNN`

`--th`

- **Type**: List of floats
- **Description**: Specifies the Theta threshold(s) for the training.
- **Required**: Yes
- **Example**: `--th 0.1 0.2 0.3`'

`--num_clients`

- **Type**: List of Integers
- **Description**: The list of client numbers for the simulations. If not provided, it defaults to the list `[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]`.
- **Example**: `--num_clients 5 10 25 50`

`--test`

- **Type**: Flag (Boolean)
- **Description**: If this flag is provided, the script will perform a bench test for the combinations.
- **Required**: No
- **Example**: `--test`

### Example
```bash
python -m utils.create_combinations --comb_id 0 --fda naive sketch --nn LeNet-5 --b 32 64 128 --e 50 --th 0.5 1.0
```

## Local simulation

### Parameters:

`--n_gpus`

- **Type**: Integer
- **Description**: Specifies the number of available GPUs in the machine. The script will distribute the processes among these GPUs in a round-robin fashion.
- **Example**: `--n_gpus 2`

`--n_proc`

- **Type**: Integer
- **Description**: The number of simulations that need to be run. See the output of the `create_combinations` script for the number of combinations generated.
- **Example**: `--n_proc 8`

`--comb_id`

- **Type**: Integer
- **Description**: The prefix of the combinations that will be read by each simulation, i.e., `.../<comb_id>.json`. The prefix for output and error log files. The generated logs will be saved in the format `c<comb_id>_proc<process_id>.out` and `c<comb_id>_proc<process_id>.err`.
- **Example**: `--comb_id 1`

`--gpu_mem`

- **Type**: Integer
- **Default**: `-1` (Indicates dynamic memory allocation by TensorFlow)
- **Description**: Specifies the GPU memory to be used. If not provided or set to `-1`, TensorFlow will dynamically allocate memory.
- **Example**: `--gpu_mem 8192`


### Example
```bash
python -m local_simulator --n_gpus 2 --n_proc 12 --comb_id 0 --gpu_mem 3584
```