import os
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Get the directory of the current script (main.py)
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path for the epoch_metrics file
epoch_metrics_path = os.path.join(script_dir, "tmp", "epoch_metrics")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json")
    parser.add_argument('--sim_id', type=int, help="The Sim ID (0, 1, 2, ...)")
    parser.add_argument('--gpu_id', type=str, help="The GPU ID that the simulation will use.")
    parser.add_argument('--gpu_mem', type=int,
                        help="The maximum GPU memory. If not provided we let TensorFlow dynamically allocate.")
    parser.add_argument('--slurm', action='store_true', help="Use if we are in SLURM HPC env.")
    args = parser.parse_args()

    if args.slurm:
        # Fix for SSL error in Aris gr.net
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        if args.gpu_mem == -1:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_mem)]
                )

    from functools import partial
    import time
    import pandas as pd

    from FdAvg.data.mnist import load_data, MNIST_N_TRAIN, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE
    from FdAvg.data.preprocessing import convert_to_tf_dataset
    from FdAvg.models.lenet5 import get_compiled_and_built_lenet
    from FdAvg.models.advanced_cnn import get_compiled_and_built_advanced_cnn
    from FdAvg.simulation.fda_simulation import single_simulation

    from FdAvg.utils.read_combinations import get_test_hyper_parameters
    from FdAvg.utils.pretty_printers import print_finish_testing_info, print_current_test_info

    hyperparameters = get_test_hyper_parameters(f'{args.comb_file_id}', args.sim_id)

    compile_and_build_model_func = None

    if hyperparameters['nn_name'] == 'AdvancedCNN':
        compile_and_build_model_func = partial(
            get_compiled_and_built_advanced_cnn, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE, 10
        )

    if hyperparameters['nn_name'] == 'LeNet-5':
        compile_and_build_model_func = partial(
            get_compiled_and_built_lenet, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE, 10
        )

    # 2. Load data
    train_dataset, test_dataset = convert_to_tf_dataset(*load_data())

    # 3. Metrics
    all_epoch_metrics = []

    # Start timer
    start_time = time.time()

    print_current_test_info(
        hyperparameters['fda_name'], hyperparameters['num_clients'],
        hyperparameters['batch_size'], hyperparameters['epochs'],
        hyperparameters['rtc_steps'], hyperparameters['nn_name'],
        hyperparameters['theta']
    )

    # 1. Naive simulation
    if hyperparameters['fda_name'] == "naive":
        epoch_metrics_with_test_id_list = single_simulation(
            fda_name="naive",
            num_clients=hyperparameters['num_clients'],
            n_train=MNIST_N_TRAIN,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=hyperparameters['batch_size'],
            num_steps_until_rtc_check=hyperparameters['rtc_steps'],
            num_epochs=hyperparameters['epochs'],
            compile_and_build_model_func=compile_and_build_model_func,
            nn_name=hyperparameters['nn_name'],
            theta=hyperparameters['theta'],
            bench_test=hyperparameters['bench_test']
        )
        all_epoch_metrics.extend(epoch_metrics_with_test_id_list)

    # 2. Linear simulation
    if hyperparameters['fda_name'] == "linear":
        epoch_metrics_with_test_id_list = single_simulation(
            fda_name="linear",
            num_clients=hyperparameters['num_clients'],
            n_train=MNIST_N_TRAIN,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=hyperparameters['batch_size'],
            num_steps_until_rtc_check=hyperparameters['rtc_steps'],
            num_epochs=hyperparameters['epochs'],
            compile_and_build_model_func=compile_and_build_model_func,
            nn_name=hyperparameters['nn_name'],
            theta=hyperparameters['theta'],
            bench_test=hyperparameters['bench_test']
        )
        all_epoch_metrics.extend(epoch_metrics_with_test_id_list)

    # 3. Sketch simulation
    if hyperparameters['fda_name'] == "sketch":
        epoch_metrics_with_test_id_list = single_simulation(
            fda_name="sketch",
            num_clients=hyperparameters['num_clients'],
            n_train=MNIST_N_TRAIN,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=hyperparameters['batch_size'],
            num_steps_until_rtc_check=hyperparameters['rtc_steps'],
            num_epochs=hyperparameters['epochs'],
            compile_and_build_model_func=compile_and_build_model_func,
            nn_name=hyperparameters['nn_name'],
            theta=hyperparameters['theta'],
            sketch_width=250,
            sketch_depth=5,
            bench_test=hyperparameters['bench_test']
        )
        all_epoch_metrics.extend(epoch_metrics_with_test_id_list)

    # 4. Synchronous simulation
    if hyperparameters['fda_name'] == "synchronous":
        epoch_metrics_with_test_id_list = single_simulation(
            fda_name="synchronous",
            num_clients=hyperparameters['num_clients'],
            n_train=MNIST_N_TRAIN,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=hyperparameters['batch_size'],
            num_steps_until_rtc_check=hyperparameters['rtc_steps'],
            num_epochs=hyperparameters['epochs'],
            compile_and_build_model_func=compile_and_build_model_func,
            nn_name=hyperparameters['nn_name'],
            bench_test=hyperparameters['bench_test']
        )
        all_epoch_metrics.extend(epoch_metrics_with_test_id_list)

    # Simulation ended
    print_finish_testing_info(
        hyperparameters['nn_name'], hyperparameters['fda_name'],
        hyperparameters['num_clients'], hyperparameters['batch_size'],
        hyperparameters['epochs'], hyperparameters['rtc_steps'],
        hyperparameters['theta'], start_time, time.time()
    )

    # Save Metrics
    epoch_metrics_df = pd.DataFrame(all_epoch_metrics)
    
    if args.slurm:
        epoch_metrics_df.to_csv(f"{epoch_metrics_path}/{hyperparameters['test_id']}.csv", index=False)
    else:
        epoch_metrics_df.to_parquet(f"{epoch_metrics_path}/{hyperparameters['test_id']}.parquet")
