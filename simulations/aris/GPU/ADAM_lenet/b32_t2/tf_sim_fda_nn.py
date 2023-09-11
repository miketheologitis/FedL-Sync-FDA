#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

slurm_localid = int(os.environ.get('SLURM_LOCALID'))

print()
print(f"slurm_localid : {slurm_localid}")

slurm_nodeid = os.environ.get('SLURM_NODEID')
print(f"Running on node : {slurm_nodeid}")

os.environ['CUDA_VISIBLE_DEVICES'] = str(slurm_localid)


import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context




import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


# ## Load EMNIST data

# In[2]:


def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    return X_train, y_train, X_test, y_test


# In[3]:


CNN_BATCH_INPUT = (None, 28, 28) # EMNIST dataset (None is used for batch size, as it varies)
CNN_INPUT_RESHAPE = (28, 28, 1)
n_train = 60_000


# ## Prepare data for Federated Learning

# In[4]:


def convert_to_tf_dataset(X_train, y_train, X_test, y_test):
    # Convert to TensorFlow Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(256)

    return train_dataset, test_dataset


# ### Slice the Tensors for each Client

# In[5]:


def create_federated_data_for_clients(num_clients, train_dataset):
    
    # Shard the data across clients CLIENT LEVEL
    client_datasets = [
        train_dataset.shard(num_clients, i)
        for i in range(num_clients)
    ]
    
    return client_datasets


# In[6]:


def prepare_federated_data_for_test(federated_data, batch_size, num_steps_until_rtc_check, seed=None):
    
    def process_client_dataset(client_dataset, batch_size, num_steps_until_rtc_check, seed, shuffle_size=512):
        return client_dataset.shuffle(shuffle_size, seed=seed).repeat().batch(batch_size)\
            .take(num_steps_until_rtc_check).prefetch(tf.data.AUTOTUNE)
        
    federated_dataset_prepared = [
        process_client_dataset(client_dataset, batch_size, num_steps_until_rtc_check, seed)
        for client_dataset in federated_data
    ]
    return federated_dataset_prepared


# # Miscallenious

# ## Variance

# In[7]:


def variance(server_cnn, client_cnns):
    
    w_t0 = server_cnn.trainable_vars_as_vector()
    
    squared_distances = [
        tf.reduce_sum(tf.square(client_cnn.trainable_vars_as_vector() - w_t0)) 
        for client_cnn in client_cnns
    ]
    
    var = tf.reduce_mean(squared_distances)
    
    return var


# ## Neural Net weights

# In[8]:


def count_weights(model):
    total_params = 0
    for layer in model.layers:
        total_params += np.sum([np.prod(weight.shape) for weight in layer.trainable_weights])
    return int(total_params)


# ## Find accuracy based on some arbitrary `compile_and_build_model_func`

# In[9]:


def current_accuracy(client_models, test_dataset, compile_and_build_model_func):
    
    tmp_model = compile_and_build_model_func()
    tmp_model.set_trainable_variables(average_client_weights(client_models))
    _, acc = tmp_model.evaluate(test_dataset, verbose=0)
    
    return acc


# ### Average NN weights

# In[10]:


def average_client_weights(client_models):
    # client_weights[0] the trainable variables of Client 0 (a list of tf.Variable)
    client_weights = [model.trainable_variables for model in client_models]

    # concise solution. per layer. `layer_weight_tensors` corresponds to a list of tensors of a layer
    avg_weights = [
        tf.reduce_mean(layer_weight_tensors, axis=0)
        for layer_weight_tensors in zip(*client_weights)
    ]

    return avg_weights


# ## Server - Clients synchronization
# 
# The assumption here is that the models have `set_trainable_variables` function implemented.

# In[11]:


def synchronize_clients(server_model, client_models):

    for client_model in client_models:
        client_model.set_trainable_variables(server_model.trainable_variables)


# ### Prepare (and restart) Client Dataset - shuffling, batching, prefetching
# 
# Proper use of `.prefetch` [explained](https://stackoverflow.com/questions/63796936/what-is-the-proper-use-of-tensorflow-dataset-prefetch-and-cache-options).
# 
# Proper ordering `.shuffle` and `.batch` and `.repeat` [explained](https://stackoverflow.com/questions/50437234/tensorflow-dataset-shuffle-then-batch-or-batch-then-shuffle)

# # LeNet-5 - Small Size (61,706 params)

# The `LeNet-5` [LeCun et al. paper from 1998](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

# In[12]:


class LeNet5(tf.keras.Model):
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        self.reshape = layers.Reshape(CNN_INPUT_RESHAPE)
        
        # Layer 1 Conv2D
        self.conv1 = layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same')
        # Layer 2 Pooling Layer
        self.avgpool1 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # Layer 3 Conv2D
        self.conv2 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')
        # Layer 4 Pooling Layer
        self.avgpool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=120, activation='tanh')
        self.dense2 = layers.Dense(units=84, activation='tanh')
        self.dense3 = layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs, training=None):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x
    
    def step(self, batch):
        
        x_batch, y_batch = batch

        with tf.GradientTape() as tape:
            # Forward pass: Compute predictions
            y_batch_pred = self(x_batch, training=True)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y_true=y_batch,
                y_pred=y_batch_pred,
                regularization_losses=self.losses
            )

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Apply gradients to the model's trainable variables (update weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics (includes the metric that tracks the loss)
        #self.compiled_metrics.update_state(y_batch, y_batch_pred)
    
    
    def train(self, dataset):

        for batch in dataset:
            self.step(batch)
            
    
    def set_trainable_variables(self, trainable_vars):
        """ Given `trainable_vars` set our `self.trainable_vars` """
        for model_var, var in zip(self.trainable_variables, trainable_vars):
            model_var.assign(var)

            
    def trainable_vars_as_vector(self):
        return tf.concat([tf.reshape(var, [-1]) for var in self.trainable_variables], axis=0)


# ### Helper function to compile and return the CNN

# **Important** function that returns a compiled and built `SimpleCNN`.

# In[13]:


def get_compiled_and_built_lenet():
    cnn = LeNet5()
    
    cnn.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # we have softmax
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')]
    )
    
    cnn.build(CNN_BATCH_INPUT)  # EMNIST dataset (None is used for batch size, as it varies)
    
    return cnn


# # Advanced Convolutional Neural Net (CNN) - Large Size

# A more complex Convolutional Neural Network with three sets of two convolutional layers, each followed by a max-pooling layer, and two dense layers with dropout for classification. Designed for 28x28 grayscale images. It has 2,592,202 weights.
# 
# Notes about `@tf.function`:
# 
# 1. After testing it might be worth it to wrap `.step` in `@tf.function`. More CPU usage. Be mindful of retracing (test it).
# 
# 2. After testing it is not worth it to wrap `.train`. Only consider `.step`.

# In[14]:


class AdvancedCNN(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        self.reshape = layers.Reshape(CNN_INPUT_RESHAPE)
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.max_pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv3 = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')
        self.conv4 = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')
        self.max_pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv5 = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')
        self.conv6 = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')
        self.max_pool3 = layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(512, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense3 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
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
    
    def step(self, batch):
        
        x_batch, y_batch = batch

        with tf.GradientTape() as tape:
            # Forward pass: Compute predictions
            y_batch_pred = self(x_batch, training=True)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y_true=y_batch,
                y_pred=y_batch_pred,
                regularization_losses=self.losses
            )

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Apply gradients to the model's trainable variables (update weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics (includes the metric that tracks the loss)
        #self.compiled_metrics.update_state(y_batch, y_batch_pred)
    
    
    def train(self, dataset):
        
        for batch in dataset:
            self.step(batch)
            
    
    def set_trainable_variables(self, trainable_vars):
        """ Given `trainable_vars` set our `self.trainable_vars` """
        for model_var, var in zip(self.trainable_variables, trainable_vars):
            model_var.assign(var)

            
    def trainable_vars_as_vector(self):
        return tf.concat([tf.reshape(var, [-1]) for var in self.trainable_variables], axis=0)
    


# ### Helper functions

# **Important** function that returns a compiled and built `AdvancedCNN`.

# In[15]:


def get_compiled_and_built_advanced_cnn():
    advanced_cnn = AdvancedCNN()
    
    advanced_cnn.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # we have softmax
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')]
    )
    
    advanced_cnn.build(CNN_BATCH_INPUT)  # EMNIST dataset (None is used for batch size, as it varies)
    
    return advanced_cnn


# # Functional Dynamic Averaging

# We follow the Functional Dynamic Averaging (FDA) scheme. Let the mean model be
# 
# $$ \overline{w_t} = \frac{1}{k} \sum_{i=1}^{k} w_t^{(i)} $$
# 
# where $ w_t^{(i)} $ is the model at time $ t $ in some round in the $i$-th learner.
# 
# Local models are trained independently and cooperatively and we want to monitor the Round Terminating Conditon (**RTC**):
# 
# $$ \frac{1}{k} \sum_{i=1}^{k} \lVert w_t^{(i)} - \overline{w_t} \rVert_2^2  \leq \Theta $$
# 
# where the left-hand side is the **model variance**, and threshold $\Theta$ is a hyperparameter of the FDA, defined at the beginning of the round; it may change at each round. When the monitoring logic cannot guarantee the validity of RTC, the round terminates. All local models are pulled into `tff.SERVER`, and $\bar{w_t}$ is set to their average. Then, another round begins.

# ### Monitoring the RTC

# FDA monitors the RTC by applying techniques from Functionary [Functional Geometric Averaging](http://users.softnet.tuc.gr/~minos/Papers/edbt19.pdf). We first restate the problem of monitoring RTC into the standard distributed stream monitoring formulation. Let
# 
# $$ S(t) =  \frac{1}{k} \sum_{i=1}^{k} S_i(t) $$
# 
# where $ S(t) \in \mathbb{R}^n $ be the "global state" of the system and $ S_i(t) \in \mathbb{R}^n $ the "local states". The goal is to monitor the threshold condition on the global state in the form $ F(S(t)) \leq \Theta $ where $ F : \mathbb{R}^n \to \mathbb{R} $ a non-linear function. Let
# 
# $$ \Delta_t^{(i)} = w_t^{(i)} - w_{t_0}^{(i)} $$
# 
# be the update at the $ i $-th learner, that is, the change to the local model at time $t$ since the beginning of the current round at time $t_0$. Let the average update be
# 
# $$ \overline{\Delta_t} = \frac{1}{k} \sum_{i=1}^{k} \Delta_t^{(i)} $$
# 
# it follows that the variance can be written as
# 
# $$ \frac{1}{k} \sum_{i=1}^{k} \lVert w_t^{(i)} - \overline{w_t} \rVert_2^2 = \Big( \frac{1}{k} \sum_{i=1}^{k} \lVert \Delta_t^{(i)} \rVert_2^2 \Big) - \lVert \overline{\Delta_t} \rVert_2^2 $$
# 
# So, conceptually, if we define
# $$ S_i(t) = \begin{bmatrix}
#            \lVert \Delta_t^{(i)} \rVert_2^2 \\
#            \Delta_t^{(i)}
#          \end{bmatrix} \quad \text{and} \quad
#          F(\begin{bmatrix}
#            v \\
#            \bf{x}
#          \end{bmatrix}) = v - \lVert \bf{x} \rVert_2^2 $$
# 
# The RTC is equivalent to condition $$ F(S(t)) \leq \Theta $$

# ## 1️⃣ Naive FDA
# 
# In the naive approach, we eliminate the update vector from the local state (i.e. recuce the dimension to 0). Define local state as
# 
# $$ S_i(t) = \lVert \Delta_t^{(i)} \rVert_2^2 \in \mathbb{R}$$ 
# 
# and the identity function
# 
# $$ F(v) = v $$
# 
# It is trivial that $ F(S(t)) \leq \Theta $ implies the RTC.

# ### Client Train
# 
# The number of steps depends on the dataset, i.e., `.take(num)` call on `tf.data.Dataset` creation!

# In[16]:


def client_train_naive(w_t0, client_cnn, client_dataset):
    """
    :param w_t0: Vector Tensor shape=(d,). Same shape with `client_cnn.trainable_vars_as_vector()`
    :return: Tensor shape=() dtype=tf.float32
    """
    
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)
    
    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0
    
    #||D(t)_i||^2 , shape = () 
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i)) # ||D(t)_i||^2
    
    return Delta_i_euc_norm_squared


# ### Train all Clients
# 
# Notes about the following potentialy general `tf.function`:
# 
# 1. Even though `clients_cnn` and `federated_dataset` contain `tf.Keras.Module` and `tf.data.Dataset` elements, they both are python lists (python side-effects). Take a look at [Looping Over Python data](https://www.tensorflow.org/guide/function#tracing) and afterwards [For example, the following loop is unrolled, even though the list contains ...](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#while-statements) to get more insight.
# 
# 2. TL;DR: It is very-very bad in terms of RAM. It produces an unrolled loop. The graph becomes consequent `Delta_i_... = ... ; S_i_clients.append(...) ;` commands `len(client_cnns)` number of times. This produces a huge graph (for instance, for `NUM_CLIENTS`=8, 4GB graph is produced). Notice that each sequence of the two commands has a big (unseen) underlying graph going to the bottom, that is, `.step` in the `tf.Keras.Module` class!
# 
# 3. Even if we had endless RAM the usage of `tf.function` is still arguable. For instance, on testing for 16 clients the difference between the two is only 20-30ms with total execution time in the order of 200-250ms. Only if we had a huge amount of CPUs or GPU we could consider it, but still... there must be a better approach (`Dask` or a different implementation).

# In[17]:


def clients_train_naive(w_t0, client_cnns, federated_dataset):
    """
    :param w_t0: Vector Tensor shape=(d,). Same shape with `client_cnns[i].trainable_vars_as_vector()`
    :return: List of `Tensor shape=() dtype=tf.float32`, one for each `client_cnn` in `client_cnns`.
    """
    
    S_i_clients = []

    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        Delta_i_euc_norm_squared = client_train_naive(w_t0, client_cnn, client_dataset)
        S_i_clients.append(Delta_i_euc_norm_squared)
    
    return S_i_clients


# ### Identity F Function

# In[18]:


def F_naive(S_i_clients):
    """ :return: Tensor shape=() dtype=tf.float32 , Naive variance approximation """
    
    S = tf.reduce_mean(S_i_clients)
    
    return S


# ## 2️⃣ Linear FDA
# 
# In the linear case, we reduce the update vector to a scalar, $ \xi \Delta_t^{(i)} \in \mathbb{R}$, where $ \xi $ is any unit vector.
# 
# Define the local state to be 
# 
# $$ S_i(t) = \begin{bmatrix}
#            \lVert \Delta_t^{(i)} \rVert_2^2 \\
#            \xi \Delta_t^{(i)}
#          \end{bmatrix} \in \mathbb{R}^2 $$
# 
# Also, define 
# 
# $$ F(v, x) = v - x^2 $$
# 
# The RTC is equivalent to condition 
# 
# $$ F(S(t)) \leq \Theta $$
# 
# A random choice of $ \xi $ is likely to perform poorly (terminate round prematurely), as it wil likely be close to orthogonal to $ \overline{\Delta_t} $. A good choice would be a vector $ \xi $ correlated to $ \overline{\Delta_t} $. A heuristic choice is to take $ \overline{\Delta_{t_0}} $ (after scaling it to norm 1), i.e., the update vector right before the current round started. All nodes can estimate this without communication, as $ \overline{w_{t_0}} - \overline{w_{t_{-1}}} $, the difference of the last two models pushed by the Server. Hence, 
# 
# $$ \xi = \frac{\overline{w_{t_0}} - \overline{w_{t_{-1}}}}{\lVert \overline{w_{t_0}} - \overline{w_{t_{-1}}} \rVert_2} $$

# In[19]:


def ksi_unit(w_t0, w_tminus1):
    """
    :param w_t0: Vector Tensor shape=(d,).
    :param w_tminus1: Vector Tensor shape=(d,)
    :return: `ξ` as defined above.
    """
    if tf.reduce_all(tf.equal(w_t0, w_tminus1)):
        # if equal then ksi becomes a random vector (will only happen in round 1)
        ksi = tf.random.normal(shape=w_t0.shape)
    else:
        ksi = w_t0 - w_tminus1

    # Normalize and return
    return tf.divide(ksi, tf.norm(ksi))


# ### Client Train
# 
# The number of steps depends on the dataset, i.e., `.take(num)` call on `tf.data.Dataset` creation!

# In[20]:


def client_train_linear(w_t0, w_tminus1, client_cnn, client_dataset):
    """
    :param w_t0: Vector Tensor shape=(d,). Same shape with `client_cnn.trainable_vars_as_vector()`
    :param w_tminus1: Vector Tensor shape=(d,). Same shape with `client_cnn.trainable_vars_as_vector()`
    :return: tuple ( Tensor shape=() dtype=tf.float32 , Tensor shape=() dtype=tf.float32 )
    """
    
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)
    
    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0
    
    #||D(t)_i||^2 , shape = () 
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i)) # ||D(t)_i||^2
    
    # heuristic unit vector ksi
    ksi = ksi_unit(w_t0, w_tminus1)
    
    # ksi * Delta_i (* is dot) , shape = ()
    ksi_Delta_i = tf.reduce_sum(tf.multiply(ksi, Delta_i))
    
    return Delta_i_euc_norm_squared, ksi_Delta_i


# ### Train all Clients
# 
# Notes about the following potentialy general `tf.function`:
# 
# 1. Even though `clients_cnn` and `federated_dataset` contain `tf.Keras.Module` and `tf.data.Dataset` elements, they both are python lists (python side-effects). Take a look at [Looping Over Python data](https://www.tensorflow.org/guide/function#tracing) and afterwards [For example, the following loop is unrolled, even though the list contains ...](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#while-statements) to get more insight.
# 
# 2. TL;DR: It is very-very bad in terms of RAM. It produces an unrolled loop. The graph becomes consequent `Delta_i_... = ... ; euc_norm_squared_clients.append(...) ; ksi_delta_clients.append(...) ;` commands `len(client_cnns)` number of times. This produces a huge graph (for instance, for `NUM_CLIENTS`=8, 4GB graph is produced). Notice that each sequence of the two commands has a big (unseen) underlying graph going to the bottom, that is, `.step` in the `tf.Keras.Module` class!
# 
# 3. Even if we had endless RAM the usage of `tf.function` is still arguable. For instance, on testing for 16 clients the difference between the two is only 20-30ms with total execution time in the order of 200-250ms. Only if we had a huge amount of CPUs or GPU we could consider it, but still... there must be a better approach (`Dask` or a different implementation).

# In[21]:


def clients_train_linear(w_t0, w_tminus1, client_cnns, federated_dataset):
    """
    :param w_t0: Vector Tensor shape=(d,). Same shape with `client_cnns[i].trainable_vars_as_vector()`
    :param w_tminus1: Vector Tensor shape=(d,). Same shape with `client_cnns[i].trainable_vars_as_vector()`
    :return: Two Lists of `Tensor shape=() dtype=tf.float32`, one for each `client_cnn` in `client_cnns`.
    """
    
    euc_norm_squared_clients = []
    ksi_delta_clients = []
    
    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):_
        Delta_i_euc_norm_squared, ksi_Delta_i = client_train_linear(
            w_t0, w_tminus1, client_cnn, client_dataset
        )

        euc_norm_squared_clients.append(Delta_i_euc_norm_squared)
        ksi_delta_clients.append(ksi_Delta_i)
    
    return euc_norm_squared_clients, ksi_delta_clients
    


# ### F Function

# In[22]:


def F_linear(euc_norm_squared_clients, ksi_delta_clients):
    """ :return: Tensor shape=() dtype=tf.float32 , Linear variance approximation """
    
    S_1 = tf.reduce_mean(euc_norm_squared_clients)
    S_2 = tf.reduce_mean(ksi_delta_clients)
    
    return S_1 - S_2**2


# ## 3️⃣ Sketch FDA

# An optimal estimator for $ \lVert \overline{\Delta_t} \rVert_2^2  $ can be obtained by employing AMS sketches. An AMS sketch of a vector $ v \in \mathbb{R}^M $ is a $ d \times m $ real matrix
# 
# $$ \Xi = \text{sk}(v) = \begin{bmatrix}
#            \Xi_1 \\
#            \Xi_2 \\
#            \vdots \\
#            \Xi_d 
#          \end{bmatrix} $$
#          
# where $ d \cdot m \ll M$. Operator sk($ \cdot $) is linear, i.e., let $a, b \in \mathbb{R}$ and $v_1, v_2 \in \mathbb{R}^N$ then 
# 
# $$ \text{sk}(a v_1 + b v_2) = a \; \text{sk}(v_1) + b \; \text{sk}(v_2)  $$
# 
# Also, sk($ v $) can be computed in $ \mathcal{O}(dN) $ steps.
# 
# The interesting property of AMS sketches is that the function 
# 
# $$ M(sk(\textbf{v})) = \underset{i=1,...,d}{\text{median}} \; \lVert \boldsymbol{\Xi}_i \rVert_2^2  $$ 
# 
# is an excellent estimator of the Euclidean norm of **v** (within relative $\epsilon$-error):
# 
# $$ M(sk(\textbf{v})) \; \in (1 \pm \epsilon) \lVert \textbf{v} \rVert_2^2 \; \; \text{with probability at least} \; (1-\delta) $$
# 
# where $m = \mathcal{O}(\frac{1}{\epsilon^2})$ and $d = \mathcal{O}(\log \frac{1}{\delta})$
# 
# Let's investigate a little further on how this helps us. The $i$-th client computes $ sk(\Delta_t^{(i)}) $ and sends it to the server. Notice
# 
# $$ M\big(sk(\Delta_t^{(1)}) + sk(\Delta_t^{(2)}) + ... + sk(\Delta_t^{(k)}) \big) = M\Big( \text{sk}\big( \sum_{i=1}^{k} \Delta_t^{(i)} \big) \Big)$$
# 
# Remember that
# 
# $$ \overline{\boldsymbol{\Delta}}_t = \frac{1}{k} \sum_{i=1}^{k} \boldsymbol{\Delta}_t^{(i)} $$
# 
# Then
#             
# $$ M\Big( \text{sk}\big( \overline{\boldsymbol{\Delta}}_t \big) \Big) = M\Big( \text{sk}\big( \frac{1}{k} \sum_{i=1}^{k} \boldsymbol{\Delta}_t^{(i)} \big) \Big) =  M\Big( \frac{1}{k} \text{sk}\big( \sum_{i=1}^{k} \boldsymbol{\Delta}_t^{(i)} \big) \Big) $$
# 
# 
# Which means that 
# 
# $$ M\Big( \frac{1}{k} \text{sk}\big( \sum_{i=1}^{k} \boldsymbol{\Delta}_t^{(i)} \big) \Big) \in (1 \pm \epsilon) \lVert \overline{\boldsymbol{\Delta}}_t \rVert_2^2 \; \; \text{w.p. at least} \; (1-\delta) $$
# 
# In the monitoring process it is essential that we do not overestimate $ \lVert \overline{\Delta_t} \rVert_2^2 $ because we would then underestimate the variance which would potentially result in actual varience exceeding $ \Theta$ without us noticing it. With this in mind,
# 
# $$ M\Big( \frac{1}{k} \text{sk}\big( \sum_{i=1}^{k} \boldsymbol{\Delta}_t^{(i)} \big) \Big) \leq (1+\epsilon) \lVert \overline{\Delta_t} \rVert_2^2 \quad \text{with probability at least} \; (1-\delta)$$
# 
# Which means
# 
# $$ \frac{1}{(1+\epsilon)} M\Big( \frac{1}{k} \text{sk}\big( \sum_{i=1}^{k} \boldsymbol{\Delta}_t^{(i)} \big) \Big) \leq \lVert \overline{\Delta_t} \rVert_2^2 \quad \text{with probability at least} \; (1-\delta)$$
# 
# Hence, the Server's estimation of $ \lVert \overline{\Delta_t} \rVert_2^2 $ is
# 
# $$ \frac{1}{(1+\epsilon)} M\Big( \frac{1}{k} \text{sk}\big( \sum_{i=1}^{k} \boldsymbol{\Delta}_t^{(i)} \big) \Big) $$
# 
# Define the local state to be 
# 
# $$ S_i(t) = \begin{bmatrix}
#            \lVert \Delta_t^{(i)} \rVert_2^2 \\
#            sk(\Delta_t^{(i)})
#          \end{bmatrix} \in \mathbb{R}^{1+d \times m} \quad \text{and} \quad
#          F(\begin{bmatrix}
#            v \\
#            \Xi
#          \end{bmatrix}) = v - \frac{1}{(1+\epsilon)}  M(\Xi) \quad \text{where} \quad \Xi = \frac{1}{k} \sum_{i=1}^{k} sk(\Delta_t^{(i)}) $$
# 
# It follows that $ F(S(t)) \leq \Theta $ implies that the variance is less or equal to $ \Theta $ with probability at least $ 1-\delta $.
# 

# ## AMS sketch

# We use `ExtensionType` which is the way to go in order to avoid unecessary graph retracing when passing around `AmsSketch` type 'objects'.

# In[23]:


class AmsSketch: 
        
    def __init__(self, depth=7, width=1500):
        self.depth = depth
        self.width = width
        self.F = tf.random.uniform(shape=(6, depth), minval=0, maxval=(1 << 31) - 1, dtype=tf.int32)

        
    def hash31(self, x, a, b):

        r = a * x + b
        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)
        return tf.bitwise.bitwise_and(fold, 2147483647)
    
    
    def tensor_hash31(self, x, a, b): # GOOD
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
        """ Extremely efficient computation of sketch with only using tensors. """
        
        print("retracing `sketch_for_vector`")
        
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
    
    
    def sketch_for_vector2(self, v):
        """ Bad implementation for tensorflow. """

        sketch = tf.zeros(shape=(self.depth, self.width), dtype=tf.float32)

        for i in tf.range(tf.shape(v)[0], dtype=tf.int32):
            pos = self.hash31(i, self.F[0], self.F[1]) % self.width
            delta = tf.cast(self.fourwise(i), dtype=tf.float32) * v[i]
            indices_to_update = tf.stack([tf.range(self.depth, dtype=tf.int32), pos], axis=1)
            sketch = tf.tensor_scatter_nd_add(sketch, indices_to_update, delta)

        return sketch
        
    
    @staticmethod
    def estimate_euc_norm_squared(sketch):

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


# ### Client Train
# 
# The number of steps depends on the dataset, i.e., `.take(num)` call on `tf.data.Dataset` creation!

# In[24]:


def client_train_sketch(w_t0, client_cnn, client_dataset, ams_sketch):
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)
    
    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0
    
    #||D(t)_i||^2 , shape = () 
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i)) # ||D(t)_i||^2 
    
    # sketch approx
    sketch = ams_sketch.sketch_for_vector(Delta_i)
    
    return Delta_i_euc_norm_squared, sketch


# ### Train all Clients
# 
# Notes about the following potentialy general `tf.function`:
# 
# 1. Even though `clients_cnn` and `federated_dataset` contain `tf.Keras.Module` and `tf.data.Dataset` elements, they both are python lists (python side-effects). Take a look at [Looping Over Python data](https://www.tensorflow.org/guide/function#tracing) and afterwards [For example, the following loop is unrolled, even though the list contains ...](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#while-statements) to get more insight.
# 
# 2. TL;DR: It is very-very bad in terms of RAM. It produces an unrolled loop. The graph becomes consequent `Delta_i_... = ... ; euc_norm_squared_clients.append(...) ; sketch_clients.append(...) ;` commands `len(client_cnns)` number of times. This produces a huge graph (for instance, for `NUM_CLIENTS`=8, 4GB graph is produced). Notice that each sequence of the two commands has a big (unseen) underlying graph going to the bottom, that is, `.step` in the `tf.Keras.Module` class!
# 
# 3. Even if we had endless RAM the usage of `tf.function` is still arguable. For instance, on testing for 16 clients the difference between the two is only 20-30ms with total execution time in the order of 200-250ms. Only if we had a huge amount of CPUs or GPU we could consider it, but still... there must be a better approach (`Dask` or a different implementation).

# In[25]:


def clients_train_sketch(w_t0, client_cnns, federated_dataset, ams_sketch):
    """
    :param w_t0: Vector Tensor shape=(d,). Same shape with `client_cnns[i].trainable_vars_as_vector()`
    :param ams_sketch: Instance of `AmsSketch` which is an extension type (no retrace) for Count-Min sketch approx.
    :return: (euc_norm_squared_clients, sketch_clients) - A list of Tensor shape=() dtype=float32 and a list of sketches
        with shape=(ams_sketch.depth, ams_sketch.width) each one corresponds to one client.
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


# ### F Function

# In[26]:


def F_sketch(euc_norm_squared_clients, sketch_clients, epsilon):
    
    S_1 = tf.reduce_mean(euc_norm_squared_clients)
    S_2 = tf.reduce_mean(sketch_clients, axis=0)  # shape=(`depth`, width`). See `Ξ` in theoretical analysis
    
    # See theoretical analysis above
    return S_1 - (1. / (1. + epsilon)) * AmsSketch.estimate_euc_norm_squared(S_2)


# # 4️⃣. Synchronous (synchronize in every step!)

# In[27]:


def clients_train_synchronous(client_cnns, federated_dataset): # NEW
    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        client_cnn.train(client_dataset)


# ### Metrics (early)

# Due to memory concerns our Metrics will consist of `namedtuple` containers which are very memory efficient.

# In[28]:


from collections import namedtuple


# In[29]:


RoundMetrics = namedtuple("RoundMetrics", ["epoch", "round", "total_fda_steps", "est_var", "actual_var"])
EpochMetrics = namedtuple("EpochMetrics", ["epoch", "total_rounds", "total_fda_steps", "accuracy"])


# A few notes about metrics:
# 
# 1. We consider the synchronization before starting training as one round, no metrics are stored we just initialize `total_rounds` to one.
# 
# 2. We consider the first Epoch to be indexed with one (not zero).
# 
# 3. `EpochMetrics` for some `epoch` correspond to final metrics of that specific epoch, that is, we store these metrics when the epoch changes to the next.
# 
# 4. In `RoundMetrics`, the column `epoch` corresponds to the epoch that this round ended at. (We can always find when it began by the previous entry).
# 
# 4. The last round will (probably) be prematurely ended by `break` because the final Epoch ended. We store this and leave it to the data analysis part to deal with it.

# ### Training Loop

# In[30]:


def federated_simulation(test_dataset, federated_dataset, fda_name, server_cnn, client_cnns, num_epochs, theta, 
                         fda_steps_in_one_epoch, compile_and_build_model_func, ams_sketch=None, epsilon=None):
    """ Run a federated learning simulation of one of the FDA methods. We keep general and time-series-like metrics. """
    
    # ---- Inits -----
    tmp_fda_steps = 0  # helper variable to monitor when Epochs pass using `fda_steps_in_one_epoch`
    epoch_count = 1
    total_rounds = 1
    total_fda_steps = 0
    est_var = 0
    
    # ----- Sync ----- TODO: Count or not first sync?
    #server_cnn.set_trainable_variables(average_client_weights(client_cnns))
    synchronize_clients(server_cnn, client_cnns)
    w_t0 = server_cnn.trainable_vars_as_vector()
    if fda_name == "linear": w_tminus1 = w_t0
    
    # ----- Metrics -----
    round_metrics_list = []
    epoch_metrics_list = []
    
    while epoch_count <= num_epochs:
        
        # We consider a `round` to be all the training until this while loop finishes and synchronization must occur
        while est_var <= theta:
            
            if fda_name == "naive":
                # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
                Delta_i_euc_norm_squared = clients_train_naive(w_t0, client_cnns, federated_dataset)
                
                # Naive estimation of variance
                est_var = F_naive(Delta_i_euc_norm_squared).numpy()
                
            if fda_name == "linear":
                # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
                euc_norm_squared_clients, ksi_delta_clients = clients_train_linear(w_t0, w_tminus1, client_cnns, federated_dataset)
                
                # Linear estimation of variance
                est_var = F_linear(euc_norm_squared_clients, ksi_delta_clients).numpy()
                
            if fda_name == "sketch":
                # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
                euc_norm_squared_clients, sketch_clients = clients_train_sketch(w_t0, client_cnns, federated_dataset, ams_sketch)
                
                # Sketch estimation of variance
                est_var = F_sketch(euc_norm_squared_clients, sketch_clients, epsilon).numpy()
            
            if fda_name == "synchronous":  # NEW
                # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
                clients_train_synchronous(client_cnns, federated_dataset)
                
                # Force synchronization in every round, `synchronous` method
                est_var = theta + 1
            
            tmp_fda_steps += 1
            total_fda_steps += 1
            
            # If Epoch has passed in this fda step
            if tmp_fda_steps >= fda_steps_in_one_epoch:
                
                # Minus here and not `tmp_fda_steps = 0` because `fda_steps_in_one_epoch` is not an integer necessarily
                # and we need to keep track of potentially more data seen in this fda step (many clients, large batch sizes)
                tmp_fda_steps -= fda_steps_in_one_epoch
                
                # ---------- Metrics ------------
                acc = current_accuracy(client_cnns, test_dataset, compile_and_build_model_func)
                epoch_metrics = EpochMetrics(epoch_count, total_rounds, total_fda_steps, acc)
                epoch_metrics_list.append(epoch_metrics)
                print(epoch_metrics) # remove
                # -------------------------------
                
                epoch_count += 1
                
                if epoch_count > num_epochs: break
        
        # Round finished

        # server average
        server_cnn.set_trainable_variables(average_client_weights(client_cnns))
        
        if fda_name == "synchronous":  # NEW
            synchronize_clients(server_cnn, client_cnns)
            est_var = 0
            total_rounds += 1
            continue

        # ------------------------- Metrics --------------------------------
        #actual_var = variance(server_cnn, client_cnns).numpy()
        #round_metrics = RoundMetrics(epoch_count, total_rounds, total_fda_steps, est_var, actual_var)
        #round_metrics_list.append(round_metrics)
        #print(round_metrics)
        # ------------------------------------------------------------------

        if fda_name == "linear": w_tminus1 = w_t0
        w_t0 = server_cnn.trainable_vars_as_vector()

        # clients sync
        synchronize_clients(server_cnn, client_cnns)
        est_var = 0

        total_rounds += 1
        
    
    return epoch_metrics_list, round_metrics_list
                


# ### Metrics
# 
# We have two types of Metrics:
# 
# 1. `EpochMetrics`: General metrics kept each Epoch (`total_rounds`, ..., `est_var`, etc.).
# 
# 2. `RoundMetrics`: Time-series type of Metrics stored in each round. This data are expected to be large in size even for small amount of tests.
# 
# We need to somehow ID every entry of both of these type of metrics since they will be combined with different tests (ex. different number of clients etc.). So, we do the logical which is pass the `TestId` as defined by the distinct combination of parameters. As explained before we choose `namedtuple` for memory efficiency.

# In[31]:


TestId = namedtuple(
        'TestId',
        ["dataset_name", "fda_name", "num_clients", "batch_size", "num_steps_until_rtc_check",
         "theta", "nn_num_weights", "sketch_width", "sketch_depth"]
)


# In[32]:


EpochMetricsWithId = namedtuple('EpochMetricsWithId', TestId._fields + EpochMetrics._fields)
RoundMetricsWithId = namedtuple('RoundMetricsWithId', TestId._fields + RoundMetrics._fields)


# Function that takes as input an instance of `TestId` and the two metrics lists that come from a single test (one FDA synchronization) and return the two lists back ready to be used as `.extend` in the general metrics lists for all the tests (properly ID'd).

# In[33]:


def process_metrics_with_test_id(epoch_metrics_list, round_metrics_list, test_id):
    
    epoch_metrics_with_test_id = [
        EpochMetricsWithId(*test_id, *epoch_metrics)
        for epoch_metrics in epoch_metrics_list
    ]
    
    round_metrics_with_test_id = [
        RoundMetricsWithId(*test_id, *round_metrics)
        for round_metrics in round_metrics_list
    ]
    
    return epoch_metrics_with_test_id, round_metrics_with_test_id


# ### Testing Preparation

# In[34]:


def prepare_for_federated_simulation(num_clients, train_dataset, batch_size, num_steps_until_rtc_check, compile_and_build_model_func, seed=None, bench_test=False):
    
    # 1. Helper variable to count Epochs
    if bench_test:
        fda_steps_in_one_epoch = 10
    else:
        fda_steps_in_one_epoch = ((n_train / batch_size) / num_clients) / num_steps_until_rtc_check
    
    # 2. Federated Dataset creation
    clients_federated_data = create_federated_data_for_clients(num_clients, train_dataset)
    federated_dataset = prepare_federated_data_for_test(clients_federated_data, batch_size, num_steps_until_rtc_check, seed)
    
    # 3. Models creation
    server_cnn = compile_and_build_model_func()
    client_cnns = [compile_and_build_model_func() for _ in range(num_clients)]
    
    return server_cnn, client_cnns, federated_dataset, fda_steps_in_one_epoch


# ### Single FDA method simulation

# In[35]:


from math import sqrt

def single_simulation(fda_name, num_clients, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
                     theta, num_epochs, sketch_width, sketch_depth, compile_and_build_model_func, bench_test=False):
    
     # 1. Preparation
    server_cnn, client_cnns, federated_dataset, fda_steps_in_one_epoch = prepare_for_federated_simulation(
        num_clients, train_dataset, batch_size, num_steps_until_rtc_check, compile_and_build_model_func, bench_test=bench_test
    )

    # 2. Simulation
    epoch_metrics_list, round_metrics_list = federated_simulation(
        test_dataset, federated_dataset, fda_name, server_cnn, client_cnns, num_epochs, theta, fda_steps_in_one_epoch,
        compile_and_build_model_func, ams_sketch = AmsSketch(width=sketch_width, depth=sketch_depth) if fda_name == "sketch" else None,
        epsilon = 1. / sqrt(sketch_width) if fda_name == "sketch" else None
    )

    # 3. Create Test ID
    test_id = TestId(
        "EMNIST", fda_name, num_clients, batch_size, num_steps_until_rtc_check, theta, 
        count_weights(server_cnn), sketch_width, sketch_depth
    )

    # 4. Store ID'd Metrics
    epoch_metrics_with_test_id_list, round_metrics_with_test_id_list = process_metrics_with_test_id(
        epoch_metrics_list, round_metrics_list, test_id
    )
    
    # Not needed, but we are proactive because `Dask` uses this
    del server_cnn, client_cnns, federated_dataset
    
    return epoch_metrics_with_test_id_list, round_metrics_with_test_id_list


# ### Print Current test Info

# In[36]:


def print_current_test_info(fda_name, num_clients, batch_size, num_epochs, num_steps_until_rtc_check, theta):
    print()
    print(f"------------ Current Test : ------------")
    print(f"FDA name : {fda_name}")
    print(f"Num Clients : {num_clients}")
    print(f"Batch size : {batch_size}")
    print(f"Num Epochs : {num_epochs}")
    print(f"Number of steps until we check RTC : {num_steps_until_rtc_check}")
    print(f"Theta : {theta}")
    print("-----------------------------------------")
    print()


# # FDA-methods simulation

# In[37]:


def fda_simulation(num_clients, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
                   theta, num_epochs, sketch_width, sketch_depth, compile_and_build_model_func, bench_test=False):
    
    complete_epoch_metrics = []
    complete_round_metrics = []
    
    for fda_name in ["naive", "linear", "sketch"]:
        
        print_current_test_info(fda_name, num_clients, batch_size, num_epochs, num_steps_until_rtc_check, theta)

        # 4. Store ID'd Metrics
        epoch_metrics_with_test_id_list, round_metrics_with_test_id_list = single_simulation(
            fda_name, num_clients, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check, theta, num_epochs,
            sketch_width if fda_name == "sketch" else -1, sketch_depth if fda_name == "sketch" else -1, 
            compile_and_build_model_func, bench_test=bench_test
        )

        complete_epoch_metrics.extend(epoch_metrics_with_test_id_list)
        complete_round_metrics.extend(round_metrics_with_test_id_list)
    
    return complete_epoch_metrics, complete_round_metrics


# # Synchronous simulation

# In[38]:


def synchronous_simulation(num_clients, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
                           theta, num_epochs, compile_and_build_model_func, bench_test=False):
    
    print_current_test_info('synchronous', num_clients, batch_size, num_epochs, num_steps_until_rtc_check, theta)
    
    return single_simulation(
            'synchronous', num_clients, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check, theta, num_epochs,
             -1, -1, compile_and_build_model_func, bench_test=bench_test
        )


# # Run tests

# In[39]:


if __name__ == '__main__':
    
    import pandas as pd
    import time
    
    import json
    
    print("GPU Setup:")
    print(tf.config.list_physical_devices('GPU'))

    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
    print()
    
    
    slurm_procid = int(os.environ.get('SLURM_PROCID'))
   
    print(f"Test procid: {slurm_procid}")
    
    with open('combinations.json', 'r') as f:
        all_combinations = json.load(f)
    
    my_comb = all_combinations[slurm_procid]

    epoch_metrics_filename = f'epoch_metrics/{my_comb["test_id"]}.csv'
    round_metrics_filename = f'round_metrics/{my_comb["test_id"]}.csv'
    
    train_dataset, test_dataset = convert_to_tf_dataset(*load_data())
    
    start_time = time.time()
    
    num_clients = my_comb['num_clients']
    batch_size = my_comb['batch_size']
    theta = my_comb['theta']
    bench_test = my_comb['bench_test']
    synchronous = my_comb['synchronous']
    
    num_epochs = my_comb['epochs']
    num_steps_until_rtc_check = my_comb['rtc_steps']
    sketch_width = 250
    sketch_depth = 5
    
    compile_and_build_model_func = get_compiled_and_built_lenet
    
    if synchronous:
        all_epoch_metrics, all_round_metrics = synchronous_simulation(
            num_clients, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
            theta, num_epochs, compile_and_build_model_func, bench_test=bench_test
        )
    else:
        all_epoch_metrics, all_round_metrics = fda_simulation(
            num_clients, train_dataset, test_dataset, batch_size, num_steps_until_rtc_check,
            theta, num_epochs, sketch_width, sketch_depth, compile_and_build_model_func, bench_test=bench_test
        )
    
    print(f"Total simulation time = {time.time()-start_time} sec")
    
    epoch_metrics_df = pd.DataFrame(all_epoch_metrics)
    round_metrics_df = pd.DataFrame(all_round_metrics)
    
    epoch_metrics_df.to_csv(epoch_metrics_filename, index=False)
    round_metrics_df.to_csv(round_metrics_filename, index=False)



