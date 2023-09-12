import tensorflow as tf


def convert_to_tf_dataset(X_train, y_train, X_test, y_test):
    """
    Convert NumPy arrays to TensorFlow datasets.

    This function takes training and testing data in the form of NumPy arrays and converts
    them into TensorFlow datasets. The test dataset is also batched with a batch size of 256.

    Args:
    - X_train (numpy.ndarray): The training data, a 3D array of shape (num_samples, 28, 28).
    - y_train (numpy.ndarray): The labels for the training data, a 1D array of shape (num_samples,).
    - X_test (numpy.ndarray): The test data, a 3D array of shape (num_samples, 28, 28).
    - y_test (numpy.ndarray): The labels for the test data, a 1D array of shape (num_samples,).

    Returns:
    - train_dataset (tf.data.Dataset): The training dataset as a TensorFlow Dataset object.
      Each element is a tuple containing:
        - A 2D tensor of shape (28, 28), representing a grayscale image.
        - A scalar tensor, representing the label of the image.
    - test_dataset (tf.data.Dataset): The test dataset as a TensorFlow Dataset object, batched with a size of 256.
      Each element is a batch containing:
        - A 3D tensor of shape (batch_size, 28, 28), representing a batch of grayscale images.
        - A 1D tensor of shape (batch_size,), representing the labels of the images in the batch.

    """
    
    # Convert to TensorFlow Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(256)

    return train_dataset, test_dataset


def prepare_federated_data(train_dataset, num_clients, batch_size, num_steps_until_rtc_check, seed=None):
    """
    Prepare federated data by sharding the original training dataset across multiple clients.

    Args:
        train_dataset (tf.data.Dataset): The original training dataset as a TensorFlow Dataset object.
            Each element is a tuple containing:
            - A 2D tensor of shape (28, 28), representing a grayscale image.
            - A scalar tensor, representing the label of the image.
        num_clients (int): The number of clients across which to shard the dataset.
        batch_size (int): The size of the batches to be used.
        num_steps_until_rtc_check (int): The number of batches to take from each client dataset.
        seed (int, optional): The random seed for shuffling the dataset. Default is None.

    Returns:
        federated_dataset_prepared (list of tf.data.Dataset): A list of prepared TensorFlow Dataset objects,
            one for each client dataset. The datasets are batched, shuffled, and prefetched.
            Each element of a prepared client dataset is a batch containing:
            - A 3D tensor of shape (batch_size, 28, 28), representing a batch of grayscale images.
            - A 1D tensor of shape (batch_size,), representing the labels of the images in the batch.

    Notes:
        - The function first shards the dataset using the `.shard` method, assigning a portion to each client.
        - It then processes each client's dataset by shuffling, batching, and prefetching.
    """
    
    def process_client_dataset(client_dataset, batch_size, num_steps_until_rtc_check, seed):
        shuffle_size = client_dataset.cardinality()  # Uniform shuffling
        return client_dataset.shuffle(shuffle_size, seed=seed).repeat().batch(batch_size)\
            .take(num_steps_until_rtc_check).prefetch(2)
    
    # Shard the data across clients CLIENT LEVEL
    clients_federated_data = [
        train_dataset.shard(num_clients, i)
        for i in range(num_clients)
    ]
        
    federated_dataset_prepared = [
        process_client_dataset(client_dataset, batch_size, num_steps_until_rtc_check, seed)
        for client_dataset in clients_federated_data
    ]
    return federated_dataset_prepared
