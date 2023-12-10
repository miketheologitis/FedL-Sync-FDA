from fdavg.data.mnist import MNIST_N_TRAIN, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE, mnist_load_federated_data
from fdavg.data.cifar10 import CIFAR10_N_TRAIN, CIFAR10_CNN_BATCH_INPUT, cifar10_load_federated_data
from fdavg.models.lenet5 import get_compiled_and_built_lenet
from fdavg.models.advanced_cnn import get_compiled_and_built_advanced_cnn
from fdavg.models.dense_net import get_compiled_and_built_densenet

from functools import partial


def derive_params(nn_name, ds_name, **kwargs):
    derived_params = {}

    # Only for MNIST dataset
    if nn_name == 'AdvancedCNN':
        derived_params['compile_and_build_model_fn'] = partial(
            get_compiled_and_built_advanced_cnn, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE, 10
        )

    if nn_name == 'LeNet-5':
        derived_params['compile_and_build_model_fn'] = partial(
            get_compiled_and_built_lenet, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE, 10
        )

    # Only for CIFAR10 dataset
    if nn_name == 'DenseNet121' or nn_name == 'DenseNet169' or nn_name == 'DenseNet201':
        derived_params['compile_and_build_model_fn'] = partial(
            get_compiled_and_built_densenet, nn_name, CIFAR10_CNN_BATCH_INPUT
        )

    if ds_name == 'MNIST':
        derived_params['load_federated_data_fn'] = mnist_load_federated_data
        derived_params['n_train'] = MNIST_N_TRAIN

    if ds_name == 'CIFAR10':
        derived_params['load_federated_data_fn'] = cifar10_load_federated_data
        derived_params['n_train'] = CIFAR10_N_TRAIN

    return derived_params
