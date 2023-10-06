import json
import os


script_directory = os.path.dirname(os.path.abspath(__file__))
# Relative path to the tmp directory
tmp_dir = '../../metrics/tmp'
comb_dir = os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/combinations'))


def get_test_hyper_parameters(comb_id, proc_id):
    with open(f'{comb_dir}/{comb_id}.json', 'r') as f:
        all_combinations = json.load(f)
    return all_combinations[proc_id]

