# Customized Operators

Public API for extended XPU operators is provided by the `itex.ops` namespace. The extended API provides better performance than the original public API.

## `itex.ops.AdamWithWeightDecayOptimizer`
This optimizer implements the Adam algorithm with weight decay
```python
itex.ops.AdamWithWeightDecayOptimizer(
    weight_decay_rate=0.001, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
    epsilon=1e-07, name='Adam',
    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"], **kwargs
)
```
This is an implementation of the AdamW optimizer described in "Decoupled Weight Decay Regularization" by Loshch ilov & Hutter ([pdf](https://arxiv.org/abs/1711.05101)). This python API `itex.ops.AdamWithWeightDecayOptimizer` replaces [tfa.optimizers.AdamW](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW).

For example:
```python
step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [10000, 15000], [1e-0, 1e-1, 1e-2])
# lr and wd can be a function or a tensor
lr = 1e-1 * schedule(step)
wd = lambda: 1e-4 * schedule(step)

# ...

optimizer = itex.ops.AdamWithWeightDecayOptimizer(learning_rate=lr, weight_decay=wd)
```

## `itex.ops.LayerNormalization`
[Layer normalization layer (Ba et al., 2016)](https://arxiv.org/abs/1607.06450).
```python
itex.ops.LayerNormalization(
    axis=-1, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    gamma_constraint=None, **kwargs
)
```
Normalize the activations of the previous layer for each given example in a batch independently, rather than across a batch like Batch Normalization. This applies a transformation that maintains the mean activation within each example close to 0, and the activation standard deviation close to 1. This python API `itex.ops.LayerNormalization` replaces [tf.keras.layers.LayerNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).

For example:
```sh
>>> import intel_extension_for_tensorflow as itex
>>> data = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
>>> layer = itex.ops.LayerNormalization(axis=1)
>>> output = layer(data, training=False)
>>> print(output)
tf.Tensor(
[[-0.99998  0.99998]
 [-0.99998  0.99998]
 [-0.99998  0.99998]
 [-0.99998  0.99998]
 [-0.99998  0.99998]], shape=(5, 2), dtype=float32)
```

## `itex.ops.gelu`
Applies the Gaussian error linear unit (GELU) activation function.
```python
itex.ops.gelu(
    features, approximate=False, name=None
)
```
Gaussian error linear unit (`GELU`) computes `x * P(X <= x)`, where `P(X) ~ N(0, 1)`. The (GELU) nonlinearity weights inputs by their value, rather than gating inputs by their sign as in `ReLU`. This python API `itex.ops.gelu` replaces [tf.nn.gelu](https://www.tensorflow.org/api_docs/python/tf/nn/gelu).

For example:
```sh
>>> import intel_extension_for_tensorflow as itex
>>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
>>> y = itex.ops.gelu(x)
>>> y.numpy()
array([-0.00404969, -0.15865526,  0.        ,  0.8413447 ,  2.9959502 ],
      dtype=float32)
>>> y = itex.ops.gelu(x, approximate=True)
>>> y.numpy()
array([-0.00363725, -0.158808  ,  0.        ,  0.841192  ,  2.9963627 ],
      dtype=float32)
```

## `itex.ops.ItexLSTM`
Long Short-Term Memory layer (first proposed in Hochreiter & Schmidhuber, 1997), this python api `itex.ops.ItexLSTM` is semantically the same as [tf.keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM).
```python
itex.ops.ItexLSTM(
    200, activation='tanh',
    recurrent_activation='sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros', **kwargs
)
```
Based on available runtime hardware and constraints, this layer will choose different implementations (ITEX-based or fallback-TensorFlow) to maximize the performance.  
If a GPU is available and all the arguments to the layer meet the requirements of the ITEX kernel (see below for details), the layer will use a fast ITEX implementation.
The requirements to use the ITEX implementation are:
  1. `activation` == `tanh`
  2. `recurrent_activation` == `sigmoid`
  3. `use_bias` is `True`
  4. Inputs, if use masking, are strictly right-padded.
  5. Eager execution is enabled in the outermost context.


For example:
```sh
>>> import intel_extension_for_tensorflow as itex
>>> inputs = tf.random.normal([32, 10, 8])
>>> lstm = itex.ops.ItexLSTM(4)
>>> output = lstm(inputs)
>>> print(output.shape)
(32, 4)
>>> lstm = itex.ops.ItexLSTM(4, return_sequences=True, return_state=True)
>>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
>>> print(whole_seq_output.shape)
(32, 10, 4)
>>> print(final_memory_state.shape)
(32, 4)
>>> print(final_carry_state.shape)
(32, 4)
```
