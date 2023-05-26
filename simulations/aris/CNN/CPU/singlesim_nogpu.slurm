#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=singlesim    # Job name
#SBATCH --output=slurm_out/ss.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=slurm_out/ss.%j.err # Stderr (%j expands to jobId)
##SBATCH --ntasks=2     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=10     # Cpus per task
#SBATCH --time=10:00   # walltime
##SBATCH --mem=16G   # memory per NODE
#SBATCH --partition=compute    # Partition
#SBATCH --account=pa230401    # Replace with your system project


## LOAD MODULES ##
module purge		# clean up loaded modules 

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


echo "Running program on Compute Island"
srun python test.py
echo "Finished program"