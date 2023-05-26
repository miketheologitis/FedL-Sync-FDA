import itertools
import json
import os


start_test_id = 0


directories = ['slurm_out', 'epoch_metrics', 'round_metrics']
# Create directories if they don't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        

epochs = 25
bench_test = False
# Define the parameter values
params = {
    "num_clients": [5, 10],
    "batch_size": [32, 64],
    "theta": [3., 4.],
    "rtc_steps": [1]
}

combinations = [
    dict(zip(params.keys(), values)) 
    for values in itertools.product(*params.values())
]

for i, combination in enumerate(combinations):
    combination["test_id"] = i + start_test_id
    combination["bench_test"] = bench_test
    combination["epochs"] = epochs
    
with open(f'combinations.json', 'w') as f:
    json.dump(combinations, f)