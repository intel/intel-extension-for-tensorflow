# Operators Override

<!Environment variable `ITEX_OPS_OVERRIDE` and>
Python api `itex.experimental_ops_override()` is provided to automatically replace some TensorFlow operators by [Custom Operators](itex_ops.md) under `itex.ops` namespace, as well as to be compatible with existing trained parameters.

# Usage
<!Once `ITEX_OPS_OVERRIDE=1` is set or >
After `itex.experimental_ops_override()` is called, these TensorFlow APIs are automatically replaced by Customized Operators. For Keras layers, their call functions will be overloaded; layer names will be kept.
<!Note that due to a known issue, users have to set `TF_NUM_INTEROP_THREADS=1` when `ITEX_OPS_OVERRIDE` is enabled to avoid possible performance drop on CPU. Calling the python API directly in model code is recommended.>

- [Layer Normalization](#layer-normalization)
- [Dense Layer](#dense-layer)
- [Gelu Activation](#gelu-activation)
- [instance Normalization](#instance-normalization)
- [LSTM](#lstm)

## Layer Normalization
`tf.keras.layers.LayerNormalization` and `keras.layers.LayerNormalization` will be fused by Customized Operators of LayerNorm and LayerNormGrad. For example:
```sh
$ python
>>> import tensorflow as tf
>>> import intel_extension_for_tensorflow as itex
>>> itex.experimental_ops_override()
>>> tf.keras.layers.LayerNormalization(
      axis=-1, epsilon=0.001, center=True, scale=True,
      beta_initializer='zeros', gamma_initializer='ones',
      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
      gamma_constraint=None, **kwargs)
>>> # it will run by op ITEXLayerNorm and ITEXLayerNormGrad
```

## Dense Layer
`tf.keras.layers.Dense` and `keras.layers.core.dense.Dense` will be optimized by BatchMatMul, BiasAdd and Activation fusion for prediction, MatMul and BiasAdd fusion for training. For example:
```sh
$ python
>>> import tensorflow as tf
>>> import intel_extension_for_tensorflow as itex
>>> itex.experimental_ops_override()
>>> tf.keras.layers.Dense(32, activation='relu')
```

## Gelu Activation
`tf.nn.gelu` will be replaced by `itex.ops.gelu`. For example:
```sh
$ python
>>> import tensorflow as tf
>>> import intel_extension_for_tensorflow as itex
>>> itex.experimental_ops_override()
>>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
>>> y = tf.nn.gelu(x)
>>> # it will run by op ITEXGelu and ITEXGeluGrad
```

## Instance Normalization
If `TensorFlow Addons` is installed, `tfa.layers.InstanceNormalization` will be replaced by custom implementation using `Transpose` and `itex.ops.LayerNormalization`. For example:
```sh
$ python
>>> import tensorflow as tf
>>> import intel_extension_for_tensorflow as itex
>>> itex.experimental_ops_override()
tfa.layers.InstanceNormalization(
      axis=-1,
      beta_initializer='zeros',
      gamma_initializer='ones',
      **kwargs)
>>> # it will run by op Transpose and ITEXLayerNorm
```

## LSTM
If Intel® Extension for TensorFlow* backend is `XPU`, `tf.keras.layers.LSTM` will be replaced by `itex.ops.ItexLSTM`. For example:
```sh
$ python
>>> import tensorflow as tf
>>> import intel_extension_for_tensorflow as itex
>>> itex.experimental_ops_override()
>>> itex.ops.ItexLSTM(
    200, activation='tanh',
    recurrent_activation='sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros', **kwargs
)
>>> # it will run by op ItexRnn
```
