from math import sqrt

from FdAvg.metrics.epoch_metrics import TestId, process_metrics_with_test_id
from FdAvg.models.miscellaneous import count_weights
from FdAvg.strategies.naive import naive_federated_simulation
from FdAvg.strategies.linear import linear_federated_simulation
from FdAvg.strategies.sketch import sketch_federated_simulation, AmsSketch
from FdAvg.strategies.synchronous import synchronous_federated_simulation
from FdAvg.strategies.gm import gm_federated_simulation


def single_simulation(ds_name, load_federated_data_fn, n_train, fda_name, num_clients, batch_size,
                      num_steps_until_rtc_check, num_epochs, compile_and_build_model_fn, nn_name, theta=0.,
                      bias=None, seed=None, bench_test=False, **kwargs):
    """
    Run a single federated learning simulation based on the given FDA method name.
    
    Args:
        ds_name (str): Name of the dataset.
        load_federated_data_fn (callable): Function to load and preprocess the federated dataset.
        n_train (int): Number of samples in the training dataset.
        fda_name (str): Name of the FDA method to use ("naive", "linear", "sketch", "synchronous").
        num_clients (int): Number of clients participating in the federated learning.
        batch_size (int): The batch size for each client.
        num_steps_until_rtc_check (int): Number of steps to perform before each RTC check.
        num_epochs (int): Number of epochs to run the simulation.
        compile_and_build_model_fn (callable): Function to compile and build the model.
        nn_name (str): Name of the neural network model.
        theta (float, optional): Variance threshold for FDA methods. Defaults to 0. for "synchronous".
        bias (float, optional): Bias parameter for the Fed dataset. Defaults to None.
        seed (int, optional): Random seed the shuffling of the Fed dataset. Defaults to None.
        bench_test (bool, optional): Whether the function is being used for a benchmark test. Defaults to False.

    Returns:
        tuple: A tuple containing two lists:
            - epoch_metrics_with_test_id_list (list): List of epoch metrics with test ID.

    Notes:
        - The function prepares the simulation environment and then runs the simulation based on the `fda_name`.
        - Metrics collected during the simulation are returned with test IDs for identification.
    """

    # 1. Helper variable to count Epochs
    if bench_test:
        fda_steps_in_one_epoch = 10
    else:
        fda_steps_in_one_epoch = ((n_train / batch_size) / num_clients) / num_steps_until_rtc_check

    # 2. Federated Dataset creation
    federated_ds, test_ds = load_federated_data_fn(
        num_clients, batch_size, num_steps_until_rtc_check, bias=bias, seed=seed
    )

    # 3. Models creation
    server_cnn = compile_and_build_model_fn()
    client_cnns = [compile_and_build_model_fn() for _ in range(num_clients)]

    epoch_metrics_list = None
    sketch_width, sketch_depth = None, None

    # 4. Simulation
    if fda_name == "naive":
        epoch_metrics_list = naive_federated_simulation(
            test_ds, federated_ds, server_cnn, client_cnns, num_epochs, theta,
            fda_steps_in_one_epoch, compile_and_build_model_fn
        )
    
    if fda_name == "linear":  
        epoch_metrics_list = linear_federated_simulation(
            test_ds, federated_ds, server_cnn, client_cnns, num_epochs, theta,
            fda_steps_in_one_epoch, compile_and_build_model_fn
        )
    
    if fda_name == "sketch":
        sketch_width, sketch_depth = 250, 5
        epoch_metrics_list = sketch_federated_simulation(
            test_ds, federated_ds, server_cnn, client_cnns, num_epochs, theta, fda_steps_in_one_epoch,
            compile_and_build_model_fn, AmsSketch(width=sketch_width, depth=sketch_depth), 1. / sqrt(sketch_width)
        )

    if fda_name == "gm":
        epoch_metrics_list = gm_federated_simulation(
            test_ds, federated_ds, server_cnn, client_cnns, num_epochs, theta,
            fda_steps_in_one_epoch, compile_and_build_model_fn
        )
        
    if fda_name == "synchronous":
        epoch_metrics_list = synchronous_federated_simulation(
            test_ds, federated_ds, server_cnn, client_cnns, num_epochs,
            fda_steps_in_one_epoch, compile_and_build_model_fn
        )

    # 5. Create Test ID
    test_id = TestId(
        ds_name, bias, fda_name, num_clients, batch_size, num_steps_until_rtc_check, theta, nn_name,
        count_weights(server_cnn), sketch_width, sketch_depth
    )

    # 6. Extend the metrics with Test ID
    epoch_metrics_with_test_id_list = process_metrics_with_test_id(
        epoch_metrics_list, test_id
    )
    
    return epoch_metrics_with_test_id_list
