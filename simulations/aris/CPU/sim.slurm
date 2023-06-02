#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=singlesim_gpu    # Job name
#SBATCH --output=slurm_out/ss.%j.out # Stdout (%j expands to jobId, %t expands to taskId)
#SBATCH --error=slurm_out/ss.%j.err  # Stderr (%j expands to jobId, %t expands to taskId)
#SBATCH --ntasks=4     # Number of tasks(processes)
#SBATCH --nodes=2     # Number of nodes requested
#SBATCH --ntasks-per-node=2     # Tasks per node
#SBATCH --cpus-per-task=10     # Threads per task
#SBATCH --time=10:00   # walltime
#SBATCH --partition=compute    # Partition
#SBATCH --account=pa230401   


## LOAD MODULES ##
module purge            # clean up loaded modules

# load necessary modules
module load gnu/8
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load tensorflow/2.4.1


echo "Running program"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"
srun python tf_sim_fda_nn.py
echo "Finished program"