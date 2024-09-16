import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
tmp_dir = '../../metrics/tmp'
local_out_path = os.path.normpath(os.path.join(script_dir, f'{tmp_dir}/local_out'))

def client_step_out_create(num_clients):
    for client_i in range(num_clients):
        with open(f"{local_out_path}/client{client_i+1}.out", "w") as file:
            pass

def client_step_out(num_clients, step, sync):
    for client_i in range(num_clients):
        with open(f"{local_out_path}/client{client_i+1}.out", "a") as file:
            sync_str = "Synchronization STARTED!" if sync else "Synchronization avoided!"
            file.write(f"Step {step:4}: {sync_str}\n")