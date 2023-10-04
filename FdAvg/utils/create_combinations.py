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
        "ds_name": args.ds_name,
        "bias": args.bias if args.bias else [None],
        "nn_name": args.nn,
        "fda_name": args.fda,
        "batch_size": args.b,
        "theta": args.th,
        "num_steps_until_rtc_check": [1],
        "num_clients": args.num_clients
    }

    combinations = [
        dict(zip(params.keys(), values)) 
        for values in itertools.product(*params.values())
    ]

    for i, combination in enumerate(combinations):
        combination["test_id"] = (
            f"{combination['nn_name'].replace('-', '')}_{combination['fda_name']}_b{combination['batch_size']}"
            f"_c{combination['num_clients']}_t{str(combination['theta']).replace('.','')}"
            f"_bias{str(combination['bias']).replace('.','')}"
        )
        combination["bench_test"] = args.test
        combination["num_epochs"] = args.e

    if not args.append_to:
        with open(f'{os.path.join(script_directory, "../tmp/combinations")}/{args.comb_file_id}.json', 'w') as f:
            json.dump(combinations, f)
            print(f"OK! Created {len(combinations)} combinations, i.e., `n_sims` = {len(combinations)}.")
    else:
        with open(f'{os.path.join(script_directory, "../tmp/combinations")}/{args.comb_file_id}.json', 'r') as f:
            old_combinations = json.load(f)
        old_combinations.extend(combinations)
        with open(f'{os.path.join(script_directory, "../tmp/combinations")}/{args.comb_file_id}.json', 'w') as f:
            json.dump(old_combinations, f)
            print(f"OK! Appended {len(combinations)} combinations, i.e., `n_sims` = {len(old_combinations)}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json", required=True)
    parser.add_argument('--ds_name', nargs='+', type=str, help="The dataset name.", default=["MNIST"])
    parser.add_argument('--bias', nargs='+', type=float, help="The bias parameter.", default=[])
    parser.add_argument('--b', nargs='+', type=int, help="The batch size(s).")
    parser.add_argument('--e', type=int, help="Number of epochs.", required=True)
    parser.add_argument('--fda', nargs='+', type=str, help="The FDA name(s).", required=True)
    parser.add_argument('--nn', nargs='+', type=str, help="The CNN name(s) ('LeNet-5' , 'AdvancedCNN')", required=True)
    parser.add_argument('--th', nargs='+', type=float, help="Theta threshold(s).", required=True)
    parser.add_argument('--num_clients', nargs='+', type=int, help="Number of clients.",
                        default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    parser.add_argument('--append_to', action='store_true',
                        help="If given, then we append to the comb file.")
    parser.add_argument('--test', action='store_true', help="If given, then we bench test.")

    create_combinations(parser.parse_args())
