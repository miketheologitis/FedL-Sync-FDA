import tensorflow as tf


class AdvancedCNN(tf.keras.Model):
    """
    Advanced Convolutional Neural Network (CNN) for image classification.

    Attributes:
    - Layers for the CNN architecture (convolutional, pooling, dense layers, dropout layers).

    Methods:
    - call: Forward pass for the model.
    - step: Compute and apply gradients for one training batch.
    - train: Train the model on a dataset.
    - set_trainable_variables: Set the trainable variables of the model.
    - trainable_vars_as_vector: Return the trainable variables as a 1D tensor.
    """
    
    def __init__(self, cnn_input_reshape, num_classes):
        """
        Initialize the advanced CNN model with given input shape and number of output classes.

        Args:
        - cnn_input_reshape (tuple): The shape to which the input should be reshaped. (e.g., (28, 28, 1))
        - num_classes (int): Number of output classes.
        """
        super(AdvancedCNN, self).__init__()
        
        self.reshape = tf.keras.layers.Reshape(cnn_input_reshape)
        
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv5 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')
        self.conv6 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')
        self.max_pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        """
        Forward pass for the model.

        Args:
        - inputs (tf.Tensor): Input tensor (batch of images).
        - training (bool, optional): Whether the forward pass is for training or inference.

        Returns:
        - tf.Tensor: Output tensor (batch of class probabilities).
        """
        x = self.reshape(inputs)  # Add a channel dimension
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        return x

    @tf.function
    def step(self, batch):
        """
        Perform one training step on a given batch of data.

        Args:
        - batch (tuple): A tuple containing two elements:
            - x_batch (tf.Tensor): A batch of input data.
            - y_batch (tf.Tensor): A batch of labels.

        This method computes the gradients using backpropagation and updates the model's trainable parameters.
        """
        x_batch, y_batch = batch

        with tf.GradientTape() as tape:
            # Forward pass: Compute predictions
            y_batch_pred = self(x_batch, training=True)

            # Compute the loss value
            loss = self.loss(y_batch, y_batch_pred)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Apply gradients to the model's trainable variables (update weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
       
    def train(self, dataset):
        """
        Train the model on an entire dataset.

        Args:
        - dataset (tf.data.Dataset): The dataset on which the model will be trained. 
        """
        for batch in dataset:
            self.step(batch)
            
    def set_trainable_variables(self, trainable_vars):
        """
        Set the model's trainable variables.

        Args:
        - trainable_vars (list of tf.Tensor): A list of tensors representing the trainable variables to be set.

        This method sets each of the model's trainable variables to the corresponding tensor in `trainable_vars`.
        """
        for model_var, var in zip(self.trainable_variables, trainable_vars):
            model_var.assign(var)

    def trainable_vars_as_vector(self):
        """
        Get the model's trainable variables as a single vector.

        Returns:
        - tf.Tensor: A 1D tensor containing all of the model's trainable variables.
        """
        return tf.concat([tf.reshape(var, [-1]) for var in self.trainable_variables], axis=0)
    

def get_compiled_and_built_advanced_cnn(cnn_batch_input, cnn_input_reshape, num_classes):
    """
    Compile and build an Advanced CNN model.

    Args:
    - cnn_batch_input (tuple): The shape of the input including batch size (e.g., (None, 28, 28)).
    - cnn_input_reshape (tuple): The shape to which the input should be reshaped (e.g., (28, 28, 1)).
    - num_classes (int): Number of output classes.

    Returns:
    - AdvancedCNN: A compiled and built Advanced CNN model.
    """
    advanced_cnn = AdvancedCNN(cnn_input_reshape, num_classes)
    
    advanced_cnn.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # we have softmax
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')]
    )
    
    advanced_cnn.build(cnn_batch_input)
    
    return advanced_cnn


# Sequential API - for reference

def sequential_advanced_cnn(cnn_input_reshape, num_classes):
    """
    Create the AdvancedCNN using the Sequential API.

    Args:
    - cnn_input_reshape (tuple): The shape to which the input should be reshaped (e.g., (28, 28, 1)).
    - num_classes (int): Number of output classes.

    Returns:
    - tf.keras.models.Sequential: An AdvancedCNN model using the Sequential API.

    Example for MNIST:
        advanced_cnn = sequential_advanced_cnn((28, 28, 1), 10)
        advanced_cnn.compile(...)
        advanced_cnn.fit(...)
    """
    return tf.keras.models.Sequential([
        # Reshape layer
        tf.keras.layers.Reshape(cnn_input_reshape, input_shape=(28, 28)),  # Example input shape, change as needed
        # First Convolutional Block
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Second Convolutional Block
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Third Convolutional Block
        tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Fully Connected Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


"""
loss = self.compiled_loss(
    y_true=y_batch,
    y_pred=y_batch_pred,
    regularization_losses=self.losses
)
"""