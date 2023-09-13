import argparse
import subprocess
import signal

processes = []


def gpu_id_gen(n_gpus):
    """
    A generator that yields the GPU ID to use.
    """
    i = 0
    while True:
        yield i % n_gpus
        i += 1


def signal_handler(signum, frame):
    """
    Signal handler to kill all child processes.
    """
    global processes  # Explicitly state that we're using the global variable
    for proc in processes:
        proc.kill()
    raise SystemExit("Killing child processes due to signal received.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int,
                        help="The number of available GPUs in this machine. (We will only use one per simulation)")
    parser.add_argument('--n_proc', type=int, help="The number of processes to run in parallel. One in each GPU.")
    parser.add_argument('--comb_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json.")
    args = parser.parse_args()

    gpu_id_generator = gpu_id_gen(args.n_gpus)

    for proc_id in range(args.n_proc):
        # Construct the command
        cmd = [
            'python', '-u', '-m', 'main_local', f'--comb_id={args.comb_id}',
            f'--proc_id={proc_id}', f'--gpu_id={next(gpu_id_generator)}'
        ]

        with open(f'tmp/local_out/c{args.comb_id}_proc{proc_id}.out', 'w') as stdout_file:
            with open(f'tmp/local_out/c{args.comb_id}_proc{proc_id}.err', 'w') as stderr_file:
                print(f"Running: {' '.join(cmd)}")
                process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)
                processes.append(process)

    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for all child processes to complete
    for p in processes:
        p.wait()

