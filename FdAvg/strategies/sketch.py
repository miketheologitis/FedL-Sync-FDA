import tensorflow as tf

from FdAvg.metrics.epoch_metrics import EpochMetrics
from FdAvg.models.miscellaneous import average_client_weights, current_accuracy, synchronize_clients


class AmsSketch:
    """ 
    AMS Sketch class for approximate second moment estimation.
    """
        
    def __init__(self, depth=7, width=1500):
        self.depth = depth
        self.width = width
        self.F = tf.random.uniform(shape=(6, depth), minval=0, maxval=(1 << 31) - 1, dtype=tf.int32)

    @staticmethod
    def hash31(x, a, b):
        r = a * x + b
        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)
        return tf.bitwise.bitwise_and(fold, 2147483647)

    @staticmethod
    def tensor_hash31(x, a, b):  # GOOD
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., tf.range(d)) """

        # Reshape x to have an extra dimension, resulting in a shape of (k, 1)
        x_reshaped = tf.expand_dims(x, axis=-1)

        # shape=(`v_dim`, 7)
        r = tf.multiply(a, x_reshaped) + b

        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)
        
        return tf.bitwise.bitwise_and(fold, 2147483647)

    def tensor_fourwise(self, x):
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., tf.range(d)) """
        # 1st use the tensor hash31
        in1 = self.tensor_hash31(x, self.F[2], self.F[3])  # shape = (`x_dim`,  `self.depth`)
        
        # 2st use the tensor hash31
        in2 = self.tensor_hash31(x, in1, self.F[4])  # shape = (`x_dim`,  `self.depth`)
        
        # 3rd use the tensor hash31
        in3 = self.tensor_hash31(x, in2, self.F[5])  # shape = (`x_dim`,  `self.depth`)
        
        in4 = tf.bitwise.bitwise_and(in3, 32768)  # shape = (`x_dim`,  `self.depth`)
        
        return 2 * (tf.bitwise.right_shift(in4, 15)) - 1  # shape = (`x_dim`,  `self.depth`)

    def fourwise(self, x):
        result = 2 * (tf.bitwise.right_shift(tf.bitwise.bitwise_and(self.hash31(self.hash31(self.hash31(x, self.F[2], self.F[3]), x, self.F[4]), x, self.F[5]), 32768), 15)) - 1
        return result

    @tf.function
    def sketch_for_vector(self, v):
        """ Extremely efficient computation of sketch with only using tensors.

        Args:
        - v (tf.Tensor): Vector to sketch. Shape=(d,).

        Returns:
        - tf.Tensor: An AMS - Sketch. Shape=(`depth`, `width`).
        """
        
        sketch = tf.zeros(shape=(self.depth, self.width), dtype=tf.float32)
        
        len_v = v.shape[0]
        
        pos_tensor = self.tensor_hash31(tf.range(len_v), self.F[0], self.F[1]) % self.width
        
        v_expand = tf.expand_dims(v, axis=-1)
        
        deltas_tensor = tf.multiply(tf.cast(self.tensor_fourwise(tf.range(len_v)), dtype=tf.float32), v_expand)
        
        range_tensor = tf.range(self.depth)
        
        # Expand dimensions to create a 2D tensor with shape (1, `self.depth`)
        range_tensor_expanded = tf.expand_dims(range_tensor, 0)

        # Use tf.tile to repeat the range `len_v` times
        repeated_range_tensor = tf.tile(range_tensor_expanded, [len_v, 1])
        
        # shape=(`len_v`, `self.depth`, 2)
        indices = tf.stack([repeated_range_tensor, pos_tensor], axis=-1)
        
        sketch = tf.tensor_scatter_nd_add(sketch, indices, deltas_tensor)
        
        return sketch

    @staticmethod
    def estimate_euc_norm_squared(sketch):
        """ Estimate the Euclidean norm squared of a vector using its AMS sketch.

        Args:
        - sketch (tf.Tensor): AMS sketch of a vector. Shape=(`depth`, `width`).

        Returns:
        - tf.Tensor: Estimated squared Euclidean norm.
        """

        def _median(v):
            """ Median of tensor `v` with shape=(n,). Note: Suboptimal O(nlogn) but it's ok bcz n = `depth`"""
            length = tf.shape(v)[0]
            sorted_v = tf.sort(v)
            middle = length // 2

            return tf.cond(
                tf.equal(length % 2, 0),
                lambda: (sorted_v[middle - 1] + sorted_v[middle]) / 2.0,
                lambda: sorted_v[middle]
            )

        return _median(tf.reduce_sum(tf.square(sketch), axis=1))
    

def client_train_sketch(w_t0, client_cnn, client_dataset, ams_sketch):
    """
    Train a client model and return the AMS sketch and square of the Euclidean norm.
    
    Args:
    - w_t0 (tf.Tensor): Initial model parameters. Shape=(d,).
    - client_cnn (object): Client CNN model.
    - client_dataset (tf.data.Dataset): Client dataset.
    - ams_sketch (AmsSketch): AMS sketch instance.
    
    Returns:
    - tuple: (Square of the Euclidean norm, AMS sketch)
    """
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)
    
    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0
    
    #  ||D(t)_i||^2 , shape = ()
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i))  # ||D(t)_i||^2
    
    # sketch approx
    sketch = ams_sketch.sketch_for_vector(Delta_i)
    
    return Delta_i_euc_norm_squared, sketch


def clients_train_sketch(w_t0, client_cnns, federated_dataset, ams_sketch):
    """
    Train multiple client models and return lists of the AMS sketches and the square of the Euclidean norms.
    
    Args:
    - w_t0 (tf.Tensor): Initial model parameters. Shape=(d,).
    - client_cnns (list): List of client CNN models.
    - federated_dataset (list): List of client datasets.
    - ams_sketch (AmsSketch): AMS sketch instance.
    
    Returns:
    - tuple: (List of squares of the Euclidean norms, List of AMS sketches)
    """
    
    euc_norm_squared_clients = []
    sketch_clients = []

    # client steps (number depends on `federated_dataset`, i.e., `.take(num)`)
    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        Delta_i_euc_norm_squared, sketch = client_train_sketch(
            w_t0, client_cnn, client_dataset, ams_sketch
        )

        euc_norm_squared_clients.append(Delta_i_euc_norm_squared)
        sketch_clients.append(sketch)
        
    return euc_norm_squared_clients, sketch_clients


def f_sketch(euc_norm_squared_clients, sketch_clients, epsilon):
    """
    Compute the approximation of the variance using sketches.
    
    Args:
    - euc_norm_squared_clients (list): List of squared Euclidean norms.
    - sketch_clients (list): List of AMS sketches.
    - epsilon (float): Error bound for the AMS sketch.
    
    Returns:
    - tf.Tensor: Approximation of the variance.
    """
    S_1 = tf.reduce_mean(euc_norm_squared_clients)
    S_2 = tf.reduce_mean(sketch_clients, axis=0)  # shape=(`depth`, width`). See `Ξ` in theoretical analysis
    
    # See theoretical analysis above
    return S_1 - (1. / (1. + epsilon)) * AmsSketch.estimate_euc_norm_squared(S_2)


def sketch_federated_simulation(test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, theta, 
                                fda_steps_in_one_epoch, compile_and_build_model_func, ams_sketch, epsilon):
    """
    Run a federated learning simulation using the AMS Sketch FDA method.
    This function collects both general and time-series-like metrics.
    
    Args:
    - test_dataset (tf.data.Dataset): The dataset for evaluating the model's performance.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for each client model.
    - server_cnn (object): The server's CNN model object.
    - client_cnns (list): A list of client CNN model objects.
    - num_epochs (int): The total number of epochs to run.
    - theta (float): The threshold for variance.
    - fda_steps_in_one_epoch (int): The number of FDA steps in one epoch.
    - compile_and_build_model_func (callable): Function to compile and build the model.
    - ams_sketch (AmsSketch): An instance of the AMS Sketch class for variance approximation.
    - epsilon (float): Error tolerance for AMS sketch.
    
    Returns:
    - epoch_metrics_list (list): A list of tuples containing epoch-wise metrics.
    
    Note:
    - An FDA step is considered as a single update from each client.
    - The function also prints metrics at the end of each epoch and round for monitoring.
    """
    
    # Initialize various counters and metrics
    tmp_fda_steps = 0  # Counter to monitor when epochs pass
    epoch_count = 1  # Epoch counter
    total_rounds = 1  # Round counter
    total_fda_steps = 0  # Total FDA steps taken
    est_var = 0  # Estimated variance

    synchronize_clients(server_cnn, client_cnns)
    
    # Initialize models and synchronize client and server models
    w_t0 = server_cnn.trainable_vars_as_vector()
    
    # Initialize list for storing metrics
    epoch_metrics_list = []
    
    while epoch_count <= num_epochs:
        
        # Continue training until estimated variance crosses the threshold
        while est_var <= theta:

            # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
            euc_norm_squared_clients, sketch_clients = clients_train_sketch(
                w_t0, client_cnns, federated_dataset, ams_sketch
            )

            # Sketch estimation of variance
            est_var = f_sketch(euc_norm_squared_clients, sketch_clients, epsilon).numpy()
            
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
                