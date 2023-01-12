import intel_extension_for_tensorflow as itex
import tensorflow as tf
import numpy as np


layer = itex.ops.LayerNormalization(axis=-1)

@tf.function
def func(x):
  y = layer(x, training=False)
  y = tf.abs(y)
  y = tf.sqrt(y)
  print(layer.beta.shape)
  print(layer.gamma.shape)
  return y

x = np.random.randn(8, 512, 1).astype(np.float32)
func(x)