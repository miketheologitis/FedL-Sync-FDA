import itertools
import json
import os

comb_id = 0

directories = ['../tmp/slurm_out', '../tmp/epoch_metrics', '../tmp/combinations']
# Create directories if they don't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
 

epochs = 15
bench_test = False
# Define the parameter values
params = {
    "nn_name": ["LeNet-5"],
    "num_clients": [4],
    "fda_name": ["synchronous"],
    "batch_size": [64],
    "theta": [0.],
    "rtc_steps": [1]
}

combinations = [
    dict(zip(params.keys(), values)) 
    for values in itertools.product(*params.values())
]


for i, combination in enumerate(combinations):
    combination["test_id"] = (
        f"{combination['nn_name'].replace('-', '')}_{combination['fda_name']}_b{combination['batch_size']}"
        f"_c{combination['num_clients']}_t{str(combination['theta']).replace('.','')}"
    )
    combination["bench_test"] = bench_test
    combination["epochs"] = epochs
    
    
with open(f'../tmp/combinations/{comb_id}.json', 'w') as f:
    json.dump(combinations, f)
