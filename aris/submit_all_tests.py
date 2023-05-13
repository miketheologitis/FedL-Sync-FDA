#!/usr/bin/env python
# coding: utf-8

# In[30]:


from string import Template
import itertools
import subprocess
import os

starting_test_id = 0

directories = ['slurm_out', 'epoch_metrics', 'round_metrics']
# Create directories if they don't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        

# Define the template
slurm_template = Template("""#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=singlesim_${test_id}
#SBATCH --output=slurm_out/ss.${test_id}.out
#SBATCH --error=slurm_out/ss.${test_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=100:00:00
#SBATCH --partition=compute
#SBATCH --account=pa230401

module purge
module load gnu/8
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load tensorflow/2.4.1

echo "Running program on Compute Island"
srun python TF_Simulation_FDA_CNN.py --test_id ${test_id} --num_clients ${num_clients} --batch_size ${batch_size} --theta ${theta} --epochs ${epochs} --rtc_steps ${rtc_steps} ${bench_test_flag}
echo "Finished program"
""")

# Define the parameter values
params = {
    "num_clients": [5, 10, 15],
    "batch_size": [32],
    "theta": [3., 4.],
    "epochs": [25],
    "rtc_steps": [1],
    "bench_test": [False]
}

combinations = list(itertools.product(*params.values()))

# Generate and submit a Slurm script for each combination of parameters
for test_id, values in enumerate(combinations):
    params_combination = dict(zip(params.keys(), values))
    
    params_combination["bench_test_flag"] = "--bench_test" if params_combination["bench_test"] else ""
    params_combination["test_id"] = test_id+starting_test_id
    
    slurm_script = slurm_template.substitute(params_combination)


    # Save the script to a file
    script_filename = f"slurm_script_{params_combination['test_id']}.slurm"
    with open(script_filename, 'w') as f:
        f.write(slurm_script)

    # Submit the script using sbatch
    subprocess.run(['sbatch', script_filename], check=True)
    
    # Delete the script file
    os.remove(script_filename)


# In[ ]:




