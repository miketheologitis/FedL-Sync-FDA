import tensorflow as tf
import numpy as np


def variance(server_model, client_models):
    """
    Compute the variance of trainable parameters between a server model and multiple client models.

    This function calculates the variance of trainable parameters between a server model and a list of
    client models. The variance is computed as the mean of the squared distances between the trainable
    parameters of each client model and the server model.

    Args:
    - server_model (object): An object representing the server model, expected to have a method `trainable_vars_as_vector`
            that returns the model's trainable parameters as a 1D tensor. In our context, for some time t, it is assumed
            that the given `server_model` is the aggregated global model from the `client_models` at time t.
    - client_models (list of objects): A list of objects representing the client models, each expected to have a
      method `trainable_vars_as_vector` that returns the model's trainable parameters as a 1D tensor.

    Returns:
    - var (tf.Tensor): A scalar tensor representing the variance of trainable parameters between the server model
      and client models.

    """
    
    w_t0 = server_model.trainable_vars_as_vector()
    
    squared_distances = [
        tf.reduce_sum(tf.square(client_model.trainable_vars_as_vector() - w_t0)) 
        for client_model in client_models
    ]
    
    var = tf.reduce_mean(squared_distances)
    
    return var


def count_weights(model):
    """
    Count the total number of trainable parameters in a Keras model.

    This function iterates through all layers of the provided Keras model and counts
    the total number of trainable parameters (weights and biases).

    Args:
    - model (keras.Model): A Keras model object. It is expected to have an attribute `layers`,
      which is a list of layers in the model.

    Returns:
    - total_params (int): The total number of trainable parameters in the model.

    Example:
    If the model has a dense layer with 128 units and input dimension of 784, and another dense
    layer with 10 units, the total number of parameters would be:
    (128 * 784) + 128 (weights and biases for the first layer) + (128 * 10) + 10 (weights and
    biases for the second layer) = 100480 + 1380 = 101860
    """
    
    total_params = 0
    for layer in model.layers:
        total_params += np.sum([np.prod(weight.shape) for weight in layer.trainable_weights])
    return int(total_params)


def current_accuracy(client_models, test_dataset, compile_and_build_model_func):
    """
    Compute the current test accuracy using averaged client model weights.

    This function takes a list of client models, averages their trainable parameters,
    and then evaluates the test accuracy on a given test dataset using these averaged parameters.

    Args:
    - client_models (list of objects): A list of objects representing the client models. Each object is expected
      to have an attribute `trainable_variables` that returns a list of `tf.Variable` objects representing the
      trainable parameters of the model.
    - test_dataset (tf.data.Dataset): A TensorFlow Dataset object representing the test data.
      Each element of the dataset is expected to be a batch containing:
        - A 3D tensor of shape (batch_size, 28, 28), representing a batch of grayscale images.
        - A 1D tensor of shape (batch_size,), representing the labels of the images in the batch.
    - compile_and_build_model_func (callable): A function with no arguments that returns a compiled Keras model.
      The returned model is expected to have a method `set_trainable_variables` that allows setting the trainable
      variables of the model, and a method `evaluate` for evaluating the model.

    Returns:
    - acc (float): The test accuracy of the model with averaged client weights on the provided test dataset.
    """
    # Create a temporary model using the provided function
    tmp_model = compile_and_build_model_func()
    # Set the trainable variables of the temporary model to the averaged client weights
    tmp_model.set_trainable_variables(average_client_weights(client_models))
    # Evaluate the temporary model on the test dataset
    _, acc = tmp_model.evaluate(test_dataset, verbose=0)
    
    return acc


def average_client_weights(client_models):
    """
    Compute the average of the trainable parameters across multiple client models.

    This function takes a list of client models and calculates the average of their 
    trainable parameters. The averaging is done layer-wise, meaning that the average 
    for each layer is computed separately and then returned as a list of average weights 
    for each layer.

    https://stackoverflow.com/questions/48212110/average-weights-in-keras-models

    Args:
    - client_models (list of objects): A list of objects representing the client models. 
      Each object is expected to have an attribute `trainable_variables` that returns a 
      list of `tf.Variable` objects representing the trainable parameters of the model.

    Returns:
    - avg_weights (list of tf.Tensor): A list of tensors representing the average weights 
      of the trainable parameters of the client models. Each tensor in the list corresponds 
      to the average weight for a specific layer.

    Example:
    If client_models[0].trainable_variables = [W1, b1, W2, b2], where W1, b1, W2, b2 are 
    tensors, then avg_weights = [avg_W1, avg_b1, avg_W2, avg_b2], where avg_W1, avg_b1, 
    avg_W2, avg_b2 are the average weights for each corresponding layer.
    """
    # Retrieve the trainable variables from each client model
    client_weights = [model.trainable_variables for model in client_models]

    # Compute the average weights for each layer
    avg_weights = [
        tf.reduce_mean(layer_weight_tensors, axis=0)
        for layer_weight_tensors in zip(*client_weights)
    ]

    return avg_weights


def synchronize_clients(server_model, client_models):
    """
    Synchronize the trainable parameters of client models with those of a server model.

    This function takes a server model and a list of client models, and sets the trainable
    parameters of each client model to be equal to those of the server model. This effectively
    synchronizes the client models with the server model.

    Args:
    - server_model (object): An object representing the server model. It is expected to have an
      attribute `trainable_variables` that returns a list of `tf.Variable` objects representing
      the trainable parameters of the model.
    - client_models (list of objects): A list of objects representing the client models. Each object
      is expected to have a method `set_trainable_variables` that allows setting the trainable
      variables of the model.
    """
    for client_model in client_models:
        client_model.set_trainable_variables(server_model.trainable_variables)


def set_trainable_variables(model, trainable_vars):
    """
    Set the model's trainable variables.

    Args:
    - model (object): An object representing the model. It is expected to have an attribute `trainable_variables`
    - trainable_vars (list of tf.Tensor): A list of tensors representing the trainable variables to be set.

    This method sets each of the model's trainable variables to the corresponding tensor in `trainable_vars`.
    """
    for model_var, var in zip(model.trainable_variables, trainable_vars):
        model_var.assign(var)


def trainable_vars_as_vector(model):
    """
    Get the model's trainable variables as a single vector.

    Returns:
    - tf.Tensor: A 1D tensor containing all of the model's trainable variables.
    """
    return tf.concat([tf.reshape(var, [-1]) for var in model.trainable_variables], axis=0)
