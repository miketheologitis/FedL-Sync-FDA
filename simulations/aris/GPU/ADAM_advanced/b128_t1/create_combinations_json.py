import itertools
import json
import os


start_test_id = 0


directories = ['slurm_out', 'epoch_metrics', 'round_metrics']
# Create directories if they don't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        

epochs = 350
bench_test = False
synchronous = False
# Define the parameter values
params = {
    "num_clients": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    "batch_size": [128],
    "theta": [1.],
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
    combination["synchronous"] = synchronous
    
with open(f'combinations.json', 'w') as f:
    json.dump(combinations, f)
