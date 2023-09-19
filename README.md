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
