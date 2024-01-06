from fdavg.data.mnist import MNIST_N_TRAIN, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE, mnist_load_federated_data
from fdavg.data.cifar10 import CIFAR10_N_TRAIN, CIFAR10_CNN_BATCH_INPUT, cifar10_load_federated_data
from fdavg.models.lenet5 import get_compiled_and_built_lenet
from fdavg.models.advanced_cnn import get_compiled_and_built_advanced_cnn
from fdavg.models.dense_net import get_compiled_and_built_densenet, create_learning_rate_schedule

from functools import partial

import tensorflow as tf


def derive_params(nn_name, ds_name, batch_size, num_clients, num_epochs, fda_name, **kwargs):
    """
    - MNIST dataset
        1. `gm`, `sketch`, `naive`, `linear` use Adam optimizer with default params on all clients, and the server has
            no optimizer at all. We put an optimizer though in order to be able to .compile the model. We do not use
            the server optimizer at all.
        2. `FedAdam` uses SGD with learning_rate 0.00316 on clients, and the server USES an optimizer. The server uses
            Adam with beta_2=0.99 and learning_rate=0.0316 utilizing the pseudo-gradient average drifts.

    - CIFAR-10
        1. `gm`, `sketch`, `naive`, `linear` use SGD with Nesterov momentum b=0.9 with a learning rate schedule starting
            from 0.1 and reducing to 0.01 and 0.001 at the 50% of epochs and 75% of epochs, respectively on all clients.
            Server does NOT use an optimizer, we just put one there so that we can .compile.
        2. `FedAvgM` uses SGD with learning_rate=0.0316 on all clients and server USES an optimizer. The server uses
            SGD with momentum b=0.9 with learning_rate=0.316.

    In this function we aim to build generic partial functions to call from `simulation.py`. We do the dirty work here
    with all the if statements. What follows is ugly...
    """

    derived_params = {}

    # Only for MNIST dataset
    if ds_name == 'MNIST':

        derived_params['load_federated_data_fn'] = mnist_load_federated_data
        derived_params['n_train'] = MNIST_N_TRAIN

        get_compiled_and_build_model = None

        if nn_name == 'AdvancedCNN':
            get_compiled_and_build_model = get_compiled_and_built_advanced_cnn
        if nn_name == 'LeNet-5':
            get_compiled_and_build_model = get_compiled_and_built_lenet

        if fda_name in ['synchronous', 'gm', 'naive', 'linear', 'sketch']:

            compile_and_build_model_fn = partial(
                get_compiled_and_build_model,
                cnn_batch_input=MNIST_CNN_BATCH_INPUT,
                cnn_input_reshape=MNIST_CNN_INPUT_RESHAPE,
                num_classes=10,
                optimizer_fn=tf.keras.optimizers.Adam
            )
            derived_params['server_compile_and_build_model_fn'] = compile_and_build_model_fn  # For .compile ONLY
            derived_params['client_compile_and_build_model_fn'] = compile_and_build_model_fn

        if fda_name == 'FedAdam':
            """
            server_optimizer_fn = partial(
                tf.keras.optimizers.Adam,
                beta_2=0.99,
                learning_rate=0.0316
            )

            
            client_optimizer_fn = partial(
                tf.keras.optimizers.SGD,
                learning_rate=0.00316
            )
            """
            server_optimizer_fn = tf.keras.optimizers.Adam
            client_optimizer_fn = tf.keras.optimizers.SGD  # deviation from paper because it works better

            derived_params['server_compile_and_build_model_fn'] = partial(
                get_compiled_and_build_model,
                cnn_batch_input=MNIST_CNN_BATCH_INPUT,
                cnn_input_reshape=MNIST_CNN_INPUT_RESHAPE,
                num_classes=10,
                optimizer_fn=server_optimizer_fn
            )

            derived_params['client_compile_and_build_model_fn'] = partial(
                get_compiled_and_build_model,
                cnn_batch_input=MNIST_CNN_BATCH_INPUT,
                cnn_input_reshape=MNIST_CNN_INPUT_RESHAPE,
                num_classes=10,
                optimizer_fn=client_optimizer_fn
            )

    if ds_name == 'CIFAR10':

        derived_params['load_federated_data_fn'] = cifar10_load_federated_data
        derived_params['n_train'] = CIFAR10_N_TRAIN

        steps_in_one_epoch = (CIFAR10_N_TRAIN / batch_size) / num_clients

        if nn_name in ['DenseNet121', 'DenseNet169', 'DenseNet201']:

            if fda_name in ['synchronous', 'gm', 'naive', 'linear', 'sketch']:

                # Specific learning rate schedule for each client.
                learning_rate_fn = partial(
                    create_learning_rate_schedule,
                    total_epochs=num_epochs,
                    steps_per_epoch=steps_in_one_epoch
                )

                optimizer_fn = partial(
                    tf.keras.optimizers.SGD,
                    momentum=0.9,
                    nesterov=True
                )

                compile_and_build_model_fn = partial(
                    get_compiled_and_built_densenet,
                    name=nn_name,
                    cnn_batch_input=CIFAR10_CNN_BATCH_INPUT,
                    learning_rate_fn=learning_rate_fn,
                    optimizer_fn=optimizer_fn
                )

                derived_params['server_compile_and_build_model_fn'] = compile_and_build_model_fn  # Only for .compile
                derived_params['client_compile_and_build_model_fn'] = compile_and_build_model_fn

        if fda_name == 'FedAvgM':

            server_optimizer_fn = partial(
                tf.keras.optimizers.SGD,
                momentum=0.9
            )

            derived_params['server_compile_and_build_model_fn'] = partial(
                get_compiled_and_built_densenet,
                name=nn_name,
                cnn_batch_input=CIFAR10_CNN_BATCH_INPUT,
                learning_rate_fn=lambda: 0.316,
                optimizer_fn=server_optimizer_fn
            )

            derived_params['client_compile_and_build_model_fn'] = partial(
                get_compiled_and_built_densenet,
                name=nn_name,
                cnn_batch_input=CIFAR10_CNN_BATCH_INPUT,
                learning_rate_fn=lambda: 0.0316,
                optimizer_fn=tf.keras.optimizers.SGD
            )

    return derived_params
