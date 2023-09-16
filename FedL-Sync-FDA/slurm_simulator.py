import argparse
import subprocess

processes = []


def gpu_id_gen(n_gpus):
    """
    A generator that yields the GPU ID to use.
    """
    i = 0
    while True:
        yield i % n_gpus
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int,
                        help="The number of available GPUs in this machine. (We will only use one per simulation)",
                        required=True)
    parser.add_argument('--n_sims', type=int, help="The number of simulations to run in parallel. One in each GPU.",
                        required=True)
    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json.",
                        required=True)
    # Add the gpu_mem argument
    parser.add_argument('--gpu_mem', type=int, default=-1,
                        help="The GPU memory to be used. If not provided we let TensorFlow dynamically allocate.")
    parser.add_argument('--sims_id', type=int,
                        help="The ID of the collective simulations we are about to run. We deploy multiple jobs per "
                             "combination file, hence, we use this unique ID to help us identify the "
                             "correct combination for this set of simulations.",
                        required=True)
    args = parser.parse_args()

    starting_sim_id = args.sims_id * args.n_sims

    gpu_id_generator = gpu_id_gen(args.n_gpus)

    for sim_id in range(starting_sim_id, starting_sim_id + args.n_sims):
        # Construct the command
        cmd = [
            'python', '-m', 'main', f'--comb_file_id={args.comb_file_id}',
            f'--sim_id={sim_id}', f'--gpu_id={next(gpu_id_generator)}',
            f'--gpu_mem={args.gpu_mem}', '--slurm'
        ]

        with open(f'tmp/local_out/c{args.comb_file_id}_sim{sim_id}.out', 'w') as stdout_file:
            with open(f'tmp/local_out/c{args.comb_file_id}_sim{sim_id}.err', 'w') as stderr_file:
                print(f"Running: {' '.join(cmd)}")
                process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)
                processes.append(process)

        print()

    print(f"The specified {args.n_sims} processes have all been started.")

    # Wait for all child processes to complete
    for p in processes:
        p.wait()

    print("All processes terminated. Exiting.")

