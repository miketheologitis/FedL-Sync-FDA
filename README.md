# Usage
Make sure you have the version of `tensorflow` and `numpy` you want to use installed. Then, install `fdavg` with

```bash
pip install fdavg
```

# API Documentation

## Table of Contents

- [Model Architectures](#model-architectures-fdavgmodels)
  - [AdvancedCNN](#advancedcnn)
  - [LeNet-5](#lenet-5)
  - [Count Weights](#count-weights)
  - [Set Trainable Variables](#set-trainable-variables)
  - [Trainable Vars as Vector](#trainable-vars-as-vector)
- [FDA Strategies](#fda-strategies-fdavgstrategies)
  - [Linear](#linear)
  - [Sketch](#sketch--amssketch)
- [Metrics](#metrics-fdavgmetrics)


## Model Architectures `fdavg.models`

### AdvancedCNN

- **Function**: `sequential_advanced_cnn(cnn_input_reshape, num_classes)`
- **Description**: Create the `AdvancedCNN` model using the Sequential API.

```python
from fdavg.models import sequential_advanced_cnn

# For MNIST
advanced_cnn = sequential_advanced_cnn(
    cnn_input_reshape=(28, 28, 1), 
    num_classes=10
)
# advanced_cnn.compile(...)
# advanced_cnn.fit(...)
```

### LeNet-5

- **Function**: `sequential_lenet5(cnn_input_reshape, num_classes)`
- **Description**: Create the `LeNet-5` model using the Sequential API.

```python
from fdavg.models import sequential_lenet5

# For MNIST
lenet5 = sequential_lenet5(
    cnn_input_reshape=(28, 28, 1), 
    num_classes=10
)
# lenet5.compile(...)
# lenet5.fit(...)
```

### Count Weights

- **Import**:
```python 
from fdavg.models import count_weights
```
- **Function**: `count_weights(model)`
- **Description**: Count the total number of trainable parameters in a Keras model.

### Set Trainable Variables
- **Import**: 
```python
from fdavg.models import set_trainable_variables
```
- **Function**: `set_trainable_variables(model, trainable_vars)`
- **Description**: Set the model's trainable variables.

### Trainable Vars as Vector
```python
from fdavg.models import trainable_vars_as_vector
```
- **Function**: `trainable_vars_as_vector(model)`
- **Description**: Get the model's trainable variables as a single vector.

## FDA strategies `fdavg.strategies`

### Linear
```python
from fdavg.strategies import ksi_unit
```
- **Function**: `ksi_unit(w_t0, w_tminus1)`
- **Description**: Calculates the heuristic unit vector ksi based on the difference of the last two models pushed by Parameter server. 

### Sketch : `AmsSketch`
```python
from fdavg.strategies import AmsSketch

ams_sketch = AmsSketch(
    depth=5,
    width=250
)

# Random vector of shape (1000,)
v = tf.random.uniform(shape=(1000,))

# Sketch vector of shape (5, 250)
sketch = ams_sketch.sketch_for_vector(v)

# Estimate the squared Euclidian norm of `v`, i.e., `||v||^2`
# using the aforementioned `sketch`
est_norm_squared = ams_sketch.estimate_euc_norm_squared(sketch)
```

# Metrics `fdavg.metrics`

### Epoch Metrics : `EpochMetrics`

This namedtuple represents metrics specific to each epoch during federated learning.

#### Attributes:
- **epoch (int)**: The epoch number.
- **total_rounds (int)**: The total number of rounds at the end of this epoch.
- **total_fda_steps (int)**: Total FDA steps taken till this epoch.
- **accuracy (float)**: Model's accuracy at the end of this epoch.

### Test-Id : `TestId`

This namedtuple represents IDs for different tests/experiments.

#### Attributes:
- **dataset_name (str)**: Name of the dataset used.
- **fda_name (str)**: FDA method name.
- **num_clients (int)**: Number of clients in the FL training.
- **batch_size (int)**: Batch size for training.
- **num_steps_until_rtc_check (int)**: Number of steps until the RTC check (usually `1`).
- **theta (float)**: The threshold (Θ) for monitoring the variance.
- **nn_num_weights (int)**: Number of weights in the neural network.
- **sketch_width (int)**: Width of the sketch. (`-1` if not applicable)
- **sketch_depth (int)**: Depth of the sketch. (`-1` if not applicable)

### Prefix all Epoch Metrics with Test-Id

- **Function**: `process_metrics_with_test_id(epoch_metrics_list, test_id)`
- **Description** : Given a list of `EpochMetrics`, i.e., `epoch_metrics_list`, and a `TestId` namedtuple, i.e., `test_id`, this function 
will prefix each `EpochMetrics` with the `test_id` and return the processed list.

#### Args:
- **epoch_metrics_list** (list of ``EpochMetrics``): List of epoch metrics.
- **test_id** (`TestId` instance): An instance of `TestId` namedtuple.

#### Returns:
- **epoch_metrics_with_test_id**: Processed list of epoch metrics with appended TestId.


## Example

```python
from fdavg.metrics import EpochMetrics, TestId, process_metrics_with_test_id
import pandas as pd

epoch_metrics_list = [EpochMetrics(...), EpochMetrics(...), ...]
test_id = TestId(...)

# Append test_id to each EpochMetrics in the list
epoch_metrics_list = process_metrics_with_test_id(epoch_metrics_list, test_id)

# Create dataframe and save it
df = pd.DataFrame(epoch_metrics_list)
df.to_parquet('metrics.parquet')
```
