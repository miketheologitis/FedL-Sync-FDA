import tensorflow as tf

from FdAvg.metrics.epoch_metrics import EpochMetrics
from FdAvg.models.miscellaneous import average_client_weights, current_accuracy, synchronize_clients


def client_train_naive(w_t0, client_cnn, client_dataset):
    """
    Trains a single client model and returns the square of the Euclidean norm of the update vector.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters as a vector. Shape=(d,).
    - client_cnn (object): The client's CNN model.
    - client_dataset (tf.data.Dataset): The dataset on which the client model will be trained.

    Returns:
    - tf.Tensor: The square of the Euclidean norm of the update vector. Shape=(), dtype=tf.float32.
    """
    
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)
    
    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0
    
    #  ||D(t)_i||^2 , shape = ()
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i))  # ||D(t)_i||^2
    
    return Delta_i_euc_norm_squared


def clients_train_naive(w_t0, client_cnns, federated_dataset):
    """
    Trains multiple client models and returns the squares of the Euclidean norms of their update vectors.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters as a vector. Shape=(d,).
    - client_cnns (list): A list of client CNN models.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for the client models.

    Returns:
    - list: A list of tensors, each representing the square of the Euclidean norm of the update vector for each client.
    """
    
    S_i_clients = []

    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        Delta_i_euc_norm_squared = client_train_naive(w_t0, client_cnn, client_dataset)
        S_i_clients.append(Delta_i_euc_norm_squared)
    
    return S_i_clients


def f_naive(S_i_clients):
    """
    Calculates the naive approximation of the variance based on the squares of the Euclidean
    norms of the update vectors.

    Args:
    - S_i_clients (list of tf.Tensor): A list of tensors, each representing the square of the
        Euclidean norm of the update vector for each client.

    Returns:
    - tf.Tensor: Naive variance approximation. Shape=(), dtype=tf.float32.
    """
    
    S = tf.reduce_mean(S_i_clients)
    
    return S


def naive_federated_simulation(test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, theta, 
                               fda_steps_in_one_epoch, compile_and_build_model_func):
    """
    Run a federated learning simulation using the Naive FDA method and collect general and time-series metrics.

    Args:
    - test_dataset (tf.data.Dataset): The dataset for evaluating the model's performance.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for the client models.
    - server_cnn (object): The server's CNN model.
    - client_cnns (list): A list of client CNN models.
    - num_epochs (int): The number of epochs to run the simulation for.
    - theta (float): The threshold for the variance of the update vectors.
    - fda_steps_in_one_epoch (int): The number of FDA steps in one epoch.
    - compile_and_build_model_func (function): Function to compile and build the model.

    Returns:
    - epoch_metrics_list (list): A list of EpochMetrics namedtuples, storing metrics per epoch.

    Note:
    - We consider an FDA step to be a single update from each client.
    - The function also outputs the metrics at the end of each epoch and round for monitoring.
    """
    
    # Initialize counters and metrics lists
    tmp_fda_steps = 0  # Counter for FDA steps within an epoch
    epoch_count = 1  # Current epoch number
    total_rounds = 1  # Total number of rounds completed
    total_fda_steps = 0  # Total number of FDA steps taken
    est_var = 0  # Estimated variance

    synchronize_clients(server_cnn, client_cnns)
    
    # Initialize models and weights
    w_t0 = server_cnn.trainable_vars_as_vector()
    
    # Initialize lists for storing metrics
    epoch_metrics_list = []
    
    while epoch_count <= num_epochs:
        
        # Continue training until estimated variance crosses the threshold
        while est_var <= theta:
            # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
            Delta_i_euc_norm_squared = clients_train_naive(w_t0, client_cnns, federated_dataset)

            # Naive estimation of variance
            est_var = f_naive(Delta_i_euc_norm_squared).numpy()
            
            tmp_fda_steps += 1
            total_fda_steps += 1
            
            # If Epoch has passed in this fda step
            if tmp_fda_steps >= fda_steps_in_one_epoch:
                
                # Minus here and not `tmp_fda_steps = 0` because `fda_steps_in_one_epoch` is not an integer necessarily
                # and we need to keep track of potentially more data seen in this fda step
                # (many clients, large batch sizes)
                tmp_fda_steps -= fda_steps_in_one_epoch
                
                # ---------- Metrics ------------
                acc = current_accuracy(client_cnns, test_dataset, compile_and_build_model_func)
                epoch_metrics = EpochMetrics(epoch_count, total_rounds, total_fda_steps, acc)
                epoch_metrics_list.append(epoch_metrics)
                print(epoch_metrics)  # remove
                # -------------------------------
                
                epoch_count += 1
                
                if epoch_count > num_epochs:
                    break
        
        # Round finished

        # server average
        server_cnn.set_trainable_variables(average_client_weights(client_cnns))

        w_t0 = server_cnn.trainable_vars_as_vector()

        # clients sync
        synchronize_clients(server_cnn, client_cnns)
        est_var = 0

        total_rounds += 1
        
    return epoch_metrics_list
                
