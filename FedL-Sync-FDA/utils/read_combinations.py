import json


def get_test_hyper_parameters(comb_filename, proc_id):
    with open(comb_filename, 'r') as f:
        all_combinations = json.load(f)
    return all_combinations[proc_id]

