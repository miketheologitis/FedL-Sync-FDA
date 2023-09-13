import itertools
import json
import os
import argparse

# Get the directory containing the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the directories relative to the script's directory
directories = [
    os.path.join(script_directory, '../tmp/slurm_out'),
    os.path.join(script_directory, '../tmp/local_out'),
    os.path.join(script_directory, '../tmp/epoch_metrics'),
    os.path.join(script_directory, '../tmp/combinations')
]
# Create directories if they don't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_combinations(args):
    # Define the parameter values
    params = {
        "nn_name": [args.nn],
        "num_clients": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "fda_name": [args.fda],
        "batch_size": [args.b],
        "theta": [args.th],
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
        combination["bench_test"] = args.test
        combination["epochs"] = args.e

    with open(f'{os.path.join(script_directory, "../tmp/combinations")}/{args.comb_id}.json', 'w') as f:
        print("OK")
        json.dump(combinations, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--comb_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json")
    parser.add_argument('--b', type=int, default=32, help="The batch size.")
    parser.add_argument('--e', type=int, default=10, help="Number of epochs.")
    parser.add_argument('--fda', type=str, default="synchronous", help="The FDA name.")
    parser.add_argument('--nn', type=str, default="LeNet-5", help="The CNN name. Either 'LeNet-5' or 'AdvancedCNN'")
    parser.add_argument('--th', type=float, default=1., help="Theta threshold.")
    parser.add_argument('--test', action='store_true', help="If given, then we bench test.")

    create_combinations(parser.parse_args())
