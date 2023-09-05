import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import pandas as pd
from functools import partial

from data import load_data, convert_to_tf_dataset, MNIST_N_TRAIN, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE
from models import get_compiled_and_built_lenet, get_compiled_and_built_advanced_cnn
from simulation import single_simulation
from utils import print_finish_testing_info, print_current_test_info


if __name__ == '__main__':
    
    # 1. Hyper-Parameters
    bench_test = True
    naive_test, linear_test, sketch_test, synchronous_test = True, True, True, True
    num_clients, batch_size, theta, num_epochs, num_steps_until_rtc_check = 5, 32, 1., 2, 1
    sketch_width, sketch_depth = 250, 5
    compile_and_build_model_func = partial(get_compiled_and_built_lenet, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE, 10)
    
    # 2. Load data
    train_dataset, test_dataset = convert_to_tf_dataset(*load_data())
    
    # 3. Metrics
    epoch_metrics_filename, round_metrics_filename = 'results/epoch_metrics.parquet', 'results/round_metrics.parquet'
    all_epoch_metrics = []
    all_round_metrics = []
    
    # Start timer
    start_time = time.time()
    
    # 1. Naive simulation
    if naive_test:
        print_current_test_info("naive", num_clients, batch_size, num_epochs, num_steps_until_rtc_check, theta)
        
        epoch_metrics_with_test_id_list, round_metrics_with_test_id_list = single_simulation(
            "naive", num_clients, MNIST_N_TRAIN, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
            num_epochs, compile_and_build_model_func, theta=theta, bench_test=bench_test
        )
        
        all_epoch_metrics.extend(epoch_metrics_with_test_id_list)
        all_round_metrics.extend(round_metrics_with_test_id_list)
    
    # 2. Linear simulation
    if linear_test:
        print_current_test_info("linear", num_clients, batch_size, num_epochs, num_steps_until_rtc_check, theta)
        
        epoch_metrics_with_test_id_list, round_metrics_with_test_id_list = single_simulation(
            "linear", num_clients, MNIST_N_TRAIN, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
            num_epochs, compile_and_build_model_func, theta=theta, bench_test=bench_test
        )
        
        all_epoch_metrics.extend(epoch_metrics_with_test_id_list)
        all_round_metrics.extend(round_metrics_with_test_id_list)
    
    # 3. Sketch simulation
    if sketch_test:
        print_current_test_info("sketch", num_clients, batch_size, num_epochs, num_steps_until_rtc_check, theta)
        
        epoch_metrics_with_test_id_list, round_metrics_with_test_id_list = single_simulation(
            "sketch", num_clients, MNIST_N_TRAIN, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
            num_epochs, compile_and_build_model_func, theta=theta, sketch_width=sketch_width,
            sketch_depth=sketch_depth, bench_test=bench_test
        )
        
        all_epoch_metrics.extend(epoch_metrics_with_test_id_list)
        all_round_metrics.extend(round_metrics_with_test_id_list)
    
    # 4. Synchronous simulation
    if synchronous_test:
        print_current_test_info("synchronous", num_clients, batch_size, num_epochs, num_steps_until_rtc_check)
        
        epoch_metrics_with_test_id_list, round_metrics_with_test_id_list = single_simulation(
            "synchronous", num_clients, MNIST_N_TRAIN, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
            num_epochs, compile_and_build_model_func, bench_test=bench_test
        )
        
        all_epoch_metrics.extend(epoch_metrics_with_test_id_list)
        all_round_metrics.extend(round_metrics_with_test_id_list)

        
    # Simulation ended
    print_finish_testing_info(
        naive_test, linear_test, sketch_test, synchronous_test, num_clients, batch_size, 
        num_epochs, num_steps_until_rtc_check, theta, start_time, time.time()
    )
    
    # Save Metrics
    epoch_metrics_df = pd.DataFrame(all_epoch_metrics)
    round_metrics_df = pd.DataFrame(all_round_metrics)
    epoch_metrics_df.to_parquet(epoch_metrics_filename)
    round_metrics_df.to_parquet(round_metrics_filename)