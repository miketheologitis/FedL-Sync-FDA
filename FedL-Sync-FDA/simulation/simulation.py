from math import sqrt

from ..data import prepare_federated_data
from ..metrics import TestId, process_metrics_with_test_id
from ..models import count_weights, synchronize_clients
from ..strategies import naive_federated_simulation, linear_federated_simulation,\
    sketch_federated_simulation, synchronous_federated_simulation, AmsSketch


def prepare_for_federated_simulation(n_train, train_dataset, num_clients, batch_size, num_steps_until_rtc_check, 
                                     compile_and_build_model_func, seed=None, bench_test=False):
    """
    Prepare the necessary components for running a federated simulation.
    
    Args:
        n_train (int): Number of samples in the training dataset.
        train_dataset (tf.data.Dataset): The training dataset.
        num_clients (int): Number of clients participating in the federated learning.
        batch_size (int): The batch size for each client.
        num_steps_until_rtc_check (int): Number of steps to perform before each RTC check.
        compile_and_build_model_func (callable): Function to compile and build the model.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        bench_test (bool, optional): Whether the function is being used for a benchmark test. Defaults to False.

    Returns:
        tuple: A tuple containing the following:
            - server_cnn (object): The server's CNN model.
            - client_cnns (list): A list of client CNN models.
            - federated_dataset (list of tf.data.Dataset): A list of preprocessed datasets for each client.
            - fda_steps_in_one_epoch (float): Number of FDA steps required to complete one epoch.

    Notes:
        - If `bench_test` is True, `fda_steps_in_one_epoch` is set to 10 for testing purposes.
        - The function creates the federated data for clients and prepares it for the test.
        - The function initializes the server and client models using `compile_and_build_model_func`.
    """
    
    # 1. Helper variable to count Epochs
    if bench_test:
        fda_steps_in_one_epoch = 10
    else:
        fda_steps_in_one_epoch = ((n_train / batch_size) / num_clients) / num_steps_until_rtc_check
    
    # 2. Federated Dataset creation
    federated_dataset = prepare_federated_data(train_dataset, num_clients, batch_size, num_steps_until_rtc_check, seed)
    
    # 3. Models creation
    server_cnn = compile_and_build_model_func()
    client_cnns = [compile_and_build_model_func() for _ in range(num_clients)]
    
    return server_cnn, client_cnns, federated_dataset, fda_steps_in_one_epoch


def single_simulation(fda_name, num_clients, n_train, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
                      num_epochs, compile_and_build_model_func, nn_name, theta=0., sketch_width=-1, sketch_depth=-1,
                      bench_test=False):
    """
    Run a single federated learning simulation based on the given FDA method name.
    
    Args:
        fda_name (str): Name of the FDA method to use ("naive", "linear", "sketch", "synchronous").
        num_clients (int): Number of clients participating in the federated learning.
        n_train (int): Number of samples in the training dataset.
        train_dataset (tf.data.Dataset): The training dataset.
        test_dataset (tf.data.Dataset): The test dataset.
        batch_size (int): The batch size for each client.
        num_steps_until_rtc_check (int): Number of steps to perform before each RTC check.
        num_epochs (int): Number of epochs to run the simulation.
        compile_and_build_model_func (callable): Function to compile and build the model.
        nn_name (str): Name of the neural network model.
        theta (float, optional): Variance threshold for FDA methods. Defaults to 0. for "synchronous".
        sketch_width (int, optional): Width parameter for the AMS sketch. Defaults to -1.
        sketch_depth (int, optional): Depth parameter for the AMS sketch. Defaults to -1.
        bench_test (bool, optional): Whether the function is being used for a benchmark test. Defaults to False.

    Returns:
        tuple: A tuple containing two lists:
            - epoch_metrics_with_test_id_list (list): List of epoch metrics with test ID.

    Notes:
        - The function prepares the simulation environment and then runs the simulation based on the `fda_name`.
        - Metrics collected during the simulation are returned with test IDs for identification.
    """
    
     # 1. Preparation
    server_cnn, client_cnns, federated_dataset, fda_steps_in_one_epoch = prepare_for_federated_simulation(
        n_train, train_dataset, num_clients, batch_size, num_steps_until_rtc_check, 
        compile_and_build_model_func, bench_test=bench_test
    )

    # 2. Simulation
    if fda_name == "naive":
        epoch_metrics_list = naive_federated_simulation(
            test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, theta, 
            fda_steps_in_one_epoch, compile_and_build_model_func
        )
    
    if fda_name == "linear":  
        epoch_metrics_list = linear_federated_simulation(
            test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, theta, 
            fda_steps_in_one_epoch, compile_and_build_model_func
        )
    
    if fda_name == "sketch":
        epoch_metrics_list = sketch_federated_simulation(
            test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, theta, fda_steps_in_one_epoch,
            compile_and_build_model_func, AmsSketch(width=sketch_width, depth=sketch_depth), 1. / sqrt(sketch_width)
        )
        
    if fda_name == "synchronous":
        epoch_metrics_list = synchronous_federated_simulation(
            test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, 
            fda_steps_in_one_epoch, compile_and_build_model_func
        )

    # 3. Create Test ID
    test_id = TestId(
        "MNIST", fda_name, num_clients, batch_size, num_steps_until_rtc_check, theta, nn_name,
        count_weights(server_cnn), sketch_width, sketch_depth
    )

    # 4. Store ID'd Metrics
    epoch_metrics_with_test_id_list = process_metrics_with_test_id(
        epoch_metrics_list, test_id
    )
    
    return epoch_metrics_with_test_id_list
