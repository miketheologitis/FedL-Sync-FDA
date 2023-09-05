#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=b256_t1    # Job name
#SBATCH --output=slurm_out/ss.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=slurm_out/ss.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=12     # Number of tasks(processes)
#SBATCH --nodes=6     # Number of nodes requested
#SBATCH --ntasks-per-node=2     # Tasks per node
#SBATCH --cpus-per-task=10     # Threads per task
#SBATCH --time=35:00:00   # walltime
#SBATCH --partition=gpu    # Partition
#SBATCH --gres=gpu:2
#SBATCH --account=pa230401    # Replace with your system project


## LOAD MODULES ##
module purge            # clean up loaded modules

# load necessary modules
#module load gnu/8
#module load intelmpi/2018
#module load python/3.8.13
#module load python/3.6.5
#module load python
#module load python/3.7.6
#module load gnu/4.9.2
#module load anaconda/5.3.1
module load gnu/8
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load tensorflow/2.4.1


echo "Running program"
echo Environment
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"
srun python tf_sim_fda_nn.py
echo "Finished program"
