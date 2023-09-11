import json

def get_test_hyper_parameters(comb_filename, proc_id):
    with open(f'../tmp/combinations/{comb_filename}.json', 'r') as f:
        all_combinations = json.load(f)
    return  all_combinations[proc_id]