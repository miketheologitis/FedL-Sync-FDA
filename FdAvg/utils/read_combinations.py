import json
import os


def get_test_hyper_parameters(comb_id, proc_id):
    # Get the directory containing the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    comb_dir = os.path.join(script_directory, '../tmp/combinations')

    with open(f'{comb_dir}/{comb_id}.json', 'r') as f:
        all_combinations = json.load(f)
    return all_combinations[proc_id]

