from tensorflow import keras

MNIST_CNN_BATCH_INPUT = (None, 28, 28)  # EMNIST dataset (None is used for batch size, as it varies)
MNIST_CNN_INPUT_RESHAPE = (28, 28, 1)
MNIST_N_TRAIN = 60_000


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
