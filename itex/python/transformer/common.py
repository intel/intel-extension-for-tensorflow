import tensorflow as tf
from tensorflow.keras import layers, initializers

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

def get_init_method(user_input, default_init_method):
  """Get initializer method for variables."""
  if user_input is None:
    return default_init_method

  if callable(user_input):
    return user_input

  assert isinstance(user_input, str)
  return initializers.get(user_input)

def get_activation_dtype(input_dtype, compute_dtype):
  assert input_dtype in [tf.float32, tf.bfloat16, tf.int8]
  assert compute_dtype in [tf.float32, tf.bfloat16]

  if input_dtype == tf.bfloat16:
    return tf.bfloat16
  return compute_dtype

def cast_if_needed(inputs, dtype):
  if inputs.dtype == tf.int8:
    return inputs
  if inputs.dtype == dtype:
    return inputs
  return tf.cast(inputs, dtype)

""" Wrapper for fp8 matmul. """
def fp8_matmul(
  inputs,
  input_scale_inv,
  input_index,
  input_fp8_dtype,
  weight,
  weight_scale_inv,
  weight_index,
  weight_fp8_dtype,
  activation_dtype,
  transpose_a=False,
  transpose_b=False,
  bias=None,
  post_add=None,
  output_amax=None,
  output_scale=None,
  output_index=None,
  output_fp8_dtype=None,
  fp8_out=False,
):
  input_shape = inputs.shape
  weight_shape = weight.shape
  batch = input_shape[1] if transpose_a else input_shape[0]
  feature = weight.shape[0] if transpose_b else weight.shape[1]
  use_bias = True
  has_post_add = True
  if bias is None:
    bias = tf.constant([], dtype=activation_dtype)
    use_bias = False
  if post_add is None:
    post_add = tf.constant([], dtype=activation_dtype)
    has_post_add = False
  
  out_dtype = activation_dtype
  if fp8_out:
    out_dtype = tf.int8
  else:
    output_amax = tf.constant([], dtype=tf.float32)
    output_scale = tf.constant([], dtype=tf.float32)
    output_fp8_dtype = ""
    output_index = -1

  output = load_ops_library.fp8_matmul(
    inputs,
    weight,
    bias=bias,
    post_add=post_add,
    a_scale_inv=input_scale_inv,
    b_scale_inv=weight_scale_inv,
    c_amax=output_amax,
    c_scale=output_scale,
    fp8_dtype_a=input_fp8_dtype,
    fp8_dtype_b=weight_fp8_dtype,
    fp8_dtype_c=output_fp8_dtype,
    transpose_a=transpose_a,
    transpose_b=transpose_b,
    fp8_meta_index_a=input_index,
    fp8_meta_index_b=weight_index,
    fp8_meta_index_c=output_index,
    use_bias=use_bias,
    has_post_add=has_post_add,
    out_dtype=out_dtype,
  )
  output = tf.reshape(output, [batch, feature])

  return output

  