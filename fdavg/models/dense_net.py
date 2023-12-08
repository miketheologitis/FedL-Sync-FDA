import tensorflow as tf
from tensorflow.keras import layers, models

""" 
Implementation from https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py

DenseNet, not pre-trained, specifically for the CIFAR-10 dataset. Preprocessing on input is assumed 
using `tensorflow.keras.applications.densenet.preprocess_input`.
"""


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    bn_axis = 3  # `channels_last` data format : (batch_size, height, width, channels)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(x.shape[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    bn_axis = 3  # `channels_last` data format : (batch_size, height, width, channels)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    x1 = layers.Dropout(0.2, name=name + '_1_dropout')(x1)  # Dropout 0.2 after convolution, Huang et al. for CIFAR-10
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x1 = layers.Dropout(0.2, name=name + '_2_dropout')(x1)  # Dropout 0.2 after convolution, Huang et al. for CIFAR-10
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def dense_net_fn(blocks, input_shape, classes):
    # Determine proper input shape
    img_input = layers.Input(shape=input_shape)

    bn_axis = 3  # `channels_last` data format : (batch_size, height, width, channels)

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='fc10')(x)

    inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = models.Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = models.Model(inputs, x, name='densenet201')
    else:
        model = models.Model(inputs, x, name='densenet')

    return model


class DenseNet:
    def __init__(self, name, input_shape=(32, 32, 3), classes=10):

        self.model = None

        if name == 'DenseNet121':
            self.model = dense_net_fn([6, 12, 24, 16], input_shape, classes)
        if name == 'DenseNet169':
            self.model = dense_net_fn([6, 12, 32, 32], input_shape, classes)
        if name == 'DenseNet201':
            self.model = dense_net_fn([6, 12, 48, 32], input_shape, classes)

    def __getattr__(self, name):
        # Automatically delegate method calls to the underlying Keras model.
        # This ensures that the custom class supports all methods of the
        # Keras model without having to define each one explicitly.
        return getattr(self.model, name)

    def step(self, batch):
        x_batch, y_batch = batch
        return self.train_on_batch(x=x_batch, y=y_batch)

    def train(self, dataset):
        for batch in dataset:
            self.step(batch)

    def set_trainable_variables(self, trainable_vars):
        for model_var, var in zip(self.trainable_variables, trainable_vars):
            model_var.assign(var)

    @tf.function
    def trainable_vars_as_vector(self):
        return tf.concat([tf.reshape(var, [-1]) for var in self.trainable_variables], axis=0)

    def per_layer_trainable_vars_as_vector(self):
        layer_vectors = [
            tf.concat([tf.reshape(var, [-1]) for var in layer.trainable_weights], axis=0)
            for layer in self.layers
            if layer.trainable_weights
        ]

        return layer_vectors

    def set_layer_weights(self, layer_i, weights):
        for model_var, var in zip(self.layers[layer_i].trainable_weights, weights):
            model_var.assign(var)

    def get_trainable_layers_indices(self):
        trainable_layers_idx = [
            i for i, layer in enumerate(self.layers)
            if layer.trainable_weights
        ]

        return trainable_layers_idx
