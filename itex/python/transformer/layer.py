import tensorflow as tf
from tensorflow.keras import layers, initializers

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from intel_extension_for_tensorflow.python.fp8.autocast import get_fp8_dtype
from intel_extension_for_tensorflow.python.transformer import BaseModule

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

class Dense(BaseModule, layers.Layer):
  def __init__(
    self,
    units,
    use_bias=True,
    gelu_activation=False,
    fp8_out=False,
    kernel_initializer=None,
    bias_initializer=None,
    **kwargs,
  ):
    super().__init__(**kwargs)

    self.units = units
    self.use_bias = use_bias
    self.gelu_activation = gelu_activation

    self.kernel_initializer = get_init_method(
      kernel_initializer, initializers.RandomNormal(mean=0.0,
                                                    stddev=0.023)
    )
    self.bias_initializer = get_init_method(
      bias_initializer, initializers.get("zeros")
    )

    # We temporarily disabled fp8 in/out feature.
    self.fp8_out = False

  def build(self, input_shape):
    """One-time allocation of the variables."""
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError(
        "The last dimension of the inputs to a Dense layer should be "
        f"defined. Found None. Full input shape received: {input_shape}"
      )

    self.kernel = self.add_weight(
      name="kernel",
      shape=(last_dim, self.units),
      initializer=self.kernel_initializer,
      trainable=True,
    )

    self.bias = None
    if self.use_bias:
      self.bias = self.add_weight(
        name="bias",
        shape=(self.units,),
        initializer=self.bias_initializer,
        trainable=True,
      )

    self.built = True

  def _fp8_dense_forward(self, inputs, kernel, bias):
    @tf.custom_gradient
    def fp8_dense_func(inputs, kernel, bias):
      self.pre_forward(training=True)
      fp8_dtype_forward = get_fp8_dtype(self.fp8_meta["recipe"], fwd=True)
      fwd_fp8_meta_tensors = self.get_fp8_meta_tensors(
        inputs, max_fp8_outs=1, fwd=True)

      if self.fp8_inp:
        inputs = load_ops_library.fp8_dequantize(
          inputs,
          fwd_fp8_meta_tensors[0][0],
          fp8_meta_index=0,
          fp8_dtype=fp8_dtype_forward,
          out_dtype=self.activation_dtype,
        )
      else:
        inputs = load_ops_library.fp8_quantize(
          inputs,
          fwd_fp8_meta_tensors[1][0],
          fwd_fp8_meta_tensors[1][1],
          fp8_meta_index=0,
          fp8_dtype=fp8_dtype_forward,
        )
        inputs = load_ops_library.fp8_dequantize(
          inputs,
          fwd_fp8_meta_tensors[1][2],
          fp8_meta_index=0,
          fp8_dtype=fp8_dtype_forward,
          out_dtype=self.activation_dtype,
        )

      kernel = load_ops_library.fp8_quantize(
        kernel,
        fwd_fp8_meta_tensors[1][0],
        fwd_fp8_meta_tensors[1][1],
        fp8_meta_index=1,
        fp8_dtype=fp8_dtype_forward,
      )
      kernel = load_ops_library.fp8_dequantize(
        kernel,
        fwd_fp8_meta_tensors[1][2],
        fp8_meta_index=1,
        fp8_dtype=fp8_dtype_forward,
        out_dtype=self.activation_dtype,
      )

      out = tf.matmul(inputs, kernel)
      if self.use_bias:
        out = out + bias

      activation = out
      if self.gelu_activation:
        activation = load_ops_library.gelu(activation, approximate=True)

      if self.fp8_out:
        activation = load_ops_library.fp8_quantize(
          activation,
          fwd_fp8_meta_tensors[2][0],
          fwd_fp8_meta_tensors[2][1],
          fp8_meta_index=0,
          fp8_dtype=fp8_dtype_forward,
        )

      def grad_fn(upstream):
        self.pre_backward()
        fp8_dtype_backward = get_fp8_dtype(self.fp8_meta["recipe"], fwd=False)

        bwd_fp8_meta_tensors = self.get_fp8_meta_tensors(
          upstream, max_fp8_outs=1, fwd=False)

        if self.fp8_out:
          upstream = load_ops_library.fp8_dequantize(
            upstream,
            bwd_fp8_meta_tensors[0][0],
            fp8_meta_index=0,
            fp8_dtype=fp8_dtype_backward,
            out_dtype=self.activation_dtype,
          )

        if self.gelu_activation:
          upstream = load_ops_library.gelu_grad(
            upstream,
            out,
            approximate=True,
          )
        bgrad = tf.reduce_sum(upstream, 0)

        if not self.fp8_out:
          upstream = load_ops_library.fp8_quantize(
            upstream,
            bwd_fp8_meta_tensors[1][0],
            bwd_fp8_meta_tensors[1][1],
            fp8_meta_index=0,
            fp8_dtype=fp8_dtype_backward,
          )
          upstream = load_ops_library.fp8_dequantize(
            upstream,
            bwd_fp8_meta_tensors[1][2],
            fp8_meta_index=0,
            fp8_dtype=fp8_dtype_backward,
            out_dtype=self.activation_dtype,
          )
        wgrad = tf.matmul(inputs, upstream, transpose_a=True)
        dgrad = tf.matmul(upstream, kernel, transpose_b=True)
        if self.fp8_inp:
          dgrad = load_ops_library.fp8_quantize(
            dgrad,
            bwd_fp8_meta_tensors[2][0],
            bwd_fp8_meta_tensors[2][1],
            fp8_meta_index=0,
            fp8_dtype=fp8_dtype_backward,
          )
        self.post_backward(dgrad)
        return dgrad, wgrad, bgrad
      return activation, grad_fn

    return fp8_dense_func(inputs, kernel, bias)

  def _get_training_value(self, training=None):
    if training is None:
      training = True
    if isinstance(training, int):
      training = bool(training)
    if not self.trainable:
      # When the layer is not trainable, it overrides the value passed
      # from model.
      training = False
    return training

  def call(self, inputs, training=None):
    is_training = self._get_training_value(training)
    self.activation_dtype = get_activation_dtype(
      inputs.dtype, self.compute_dtype)

    self.fp8_inp = inputs.dtype == tf.int8
    num_fp8_inps = 1 if self.fp8_inp else 0
    num_fp8_outs = 1 if self.fp8_out else 0
    self.fp8_init(
      num_fp8_inps=num_fp8_inps, num_gemms=1, num_fp8_outs=num_fp8_outs)

    inputs = cast_if_needed(inputs, self.activation_dtype)
    kernel = cast_if_needed(self.kernel, self.activation_dtype)
    bias = self.bias
    if self.use_bias:
      bias = cast_if_needed(bias, self.activation_dtype)

    if self.fp8:
      assert is_training
      out = self._fp8_dense_forward(inputs, kernel, bias)
      self.post_forward(out)
    else:
      out = tf.matmul(inputs, kernel)
      if self.use_bias:
        out = out + bias
      if self.gelu_activation:
        out = load_ops_library.gelu(out, approximate=True)
    return out
