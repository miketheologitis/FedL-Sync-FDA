import tensorflow as tf
from tensorflow import keras

def load_data():
    """
    Load the MNIST dataset and normalize the pixel values.

    This function loads the MNIST dataset using Keras's built-in dataset API.
    It normalizes the pixel values of the images by dividing them by 255.0.
    
    Returns:
    - X_train (numpy.ndarray): The training data, a 3D array of shape (num_samples, 28, 28).
    - y_train (numpy.ndarray): The labels for the training data, a 1D array of shape (num_samples,).
    - X_test (numpy.ndarray): The test data, a 3D array of shape (num_samples, 28, 28).
    - y_test (numpy.ndarray): The labels for the test data, a 1D array of shape (num_samples,).
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    return X_train, y_train, X_test, y_test


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


def create_federated_data_for_clients(num_clients, train_dataset):
    """
    Create federated datasets by sharding the original training dataset across multiple clients.

    This function takes the original training dataset and the number of clients, and shards the
    dataset across the specified number of clients. Each shard is a TensorFlow Dataset object
    and will be assigned to one client.

    Args:
    - num_clients (int): The number of clients across which to shard the dataset.
    - train_dataset (tf.data.Dataset): The original training dataset as a TensorFlow Dataset object.
      Each element is a tuple containing:
        - A 2D tensor of shape (28, 28), representing a grayscale image.
        - A scalar tensor, representing the label of the image.

    Returns:
    - client_datasets (list of tf.data.Dataset): A list of TensorFlow Dataset objects, one for each client.
      Each client dataset is a shard of the original `train_dataset`.
      Each element of a client dataset is a tuple containing:
        - A 2D tensor of shape (28, 28), representing a grayscale image.
        - A scalar tensor, representing the label of the image.

    """
    # Shard the data across clients CLIENT LEVEL
    client_datasets = [
        train_dataset.shard(num_clients, i)
        for i in range(num_clients)
    ]
    
    return client_datasets


def prepare_federated_data_for_test(federated_data, batch_size, num_steps_until_rtc_check, seed=None):
    """
    Prepare federated datasets for test by batching, shuffling, and prefetching.

    This function takes a list of federated datasets and applies several preprocessing steps:
    shuffling, repeating, batching, taking a fixed number of steps, and prefetching.

    Args:
    - federated_data (list of tf.data.Dataset): A list of TensorFlow Dataset objects representing federated data.
      Each element of a client dataset is a tuple containing:
        - A 2D tensor of shape (28, 28), representing a grayscale image.
        - A scalar tensor, representing the label of the image.
    - batch_size (int): The size of the batches to be used.
    - num_steps_until_rtc_check (int): The number of batches to take from each client dataset.
    - seed (int, optional): The random seed for shuffling the dataset. Default is None.

    Returns:
    - federated_dataset_prepared (list of tf.data.Dataset): A list of prepared TensorFlow Dataset objects,
      one for each client dataset. The datasets are batched, shuffled, and prefetched.
      Each element of a prepared client dataset is a batch containing:
        - A 3D tensor of shape (batch_size, 28, 28), representing a batch of grayscale images.
        - A 1D tensor of shape (batch_size,), representing the labels of the images in the batch.

    """
    
    def process_client_dataset(client_dataset, batch_size, num_steps_until_rtc_check, seed, shuffle_size=512):
        return client_dataset.shuffle(shuffle_size, seed=seed).repeat().batch(batch_size)\
            .take(num_steps_until_rtc_check).prefetch(tf.data.AUTOTUNE)
        
    federated_dataset_prepared = [
        process_client_dataset(client_dataset, batch_size, num_steps_until_rtc_check, seed)
        for client_dataset in federated_data
    ]
    return federated_dataset_prepared