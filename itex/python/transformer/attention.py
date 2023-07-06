import tensorflow as tf
from tensorflow.keras import layers, initializers
from tensorflow.python.ops import random_ops

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from intel_extension_for_tensorflow.python.fp8.autocast import get_fp8_dtype
from intel_extension_for_tensorflow.python.transformer import BaseModule
from intel_extension_for_tensorflow.python.transformer.common import (
  get_init_method,
  get_activation_dtype,
  cast_if_needed,
  fp8_matmul,
)

fp8_scaled_dot_product_attention = (
  load_ops_library.fp8_scaled_dot_product_attention
)
fp8_scaled_dot_product_attention_grad = (
  load_ops_library.fp8_scaled_dot_product_attention_grad
)

""" Only support self-attention now. """
class MultiHeadAttention(BaseModule, layers.Layer):
  def __init__(
    self,
    hidden_size,
    head_size,
    attention_dropout=0.1,
    init_method=None,
    output_layer_init_method=None,
    attention_type="self",
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.hidden_size = hidden_size
    assert attention_type == "self"
    self.attention_type = attention_type
    self.init_method = init_method
    self.output_layer_init_method = output_layer_init_method
    self.bias_initializer = initializers.get("zeros")

    self.head_size = head_size
    self.num_attention_heads = int(hidden_size / head_size)
    self.attention_dropout = attention_dropout

    self.default_initializer = get_init_method(
      init_method,
      initializers.RandomNormal(mean=0.0, stddev=0.023),
    )
    self.output_initializer = get_init_method(
      output_layer_init_method,
      initializers.RandomNormal(mean=0.0, stddev=0.023),
    )

    self._autocast = False

  def build(self, input_shape):
    """One-time allocation of the variables."""
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.inputs_feature = last_dim

    self.query_kernel = self.add_weight(
      name="query_kernel",
      shape=(last_dim, self.hidden_size),
      initializer=self.default_initializer,
      trainable=True,
    )
    self.query_bias = self.add_weight(
      name="query_bias",
      shape=(self.hidden_size),
      initializer=self.bias_initializer,
      trainable=True,
    )

    self.key_kernel = self.add_weight(
      name="key_kernel",
      shape=(last_dim, self.hidden_size),
      initializer=self.default_initializer,
      trainable=True,
    )
    self.key_bias = self.add_weight(
      name="key_bias",
      shape=(self.hidden_size),
      initializer=self.bias_initializer,
      trainable=True,
    )

    self.value_kernel = self.add_weight(
      name="value_kernel",
      shape=(last_dim, self.hidden_size),
      initializer=self.default_initializer,
      trainable=True,
    )
    self.value_bias = self.add_weight(
      name="value_bias",
      shape=(self.hidden_size),
      initializer=self.bias_initializer,
      trainable=True,
    )

    self.output_kernel = self.add_weight(
      name="output_kernel",
      shape=(self.hidden_size, self.hidden_size),
      initializer=self.output_initializer,
      trainable=True,
    )
    self.output_bias = self.add_weight(
      name="output_bias",
      shape=(self.hidden_size),
      initializer=self.bias_initializer,
      trainable=True,
    )

    self.built = True
 
  def _fp8_mha_forward(
    self,
    inputs,
    query_kernel,
    query_bias,
    key_kernel,
    key_bias,
    value_kernel,
    value_bias,
    output_kernel,
    output_bias,
    qk_scale,
    attention_mask,
    dropout_mask,
  ):

    @tf.custom_gradient
    def fp8_mha_func(
      inputs,
      query_kernel,
      query_bias,
      key_kernel,
      key_bias,
      value_kernel,
      value_bias,
      output_kernel,
      output_bias,
      qk_scale,
      attention_mask,
      dropout_mask,
    ):
      self.pre_forward(training=True)
      fp8_dtype_forward = get_fp8_dtype(self.fp8_meta["recipe"], fwd=True)
      fwd_fp8_meta_tensors = self.get_fp8_meta_tensors(
        inputs, max_fp8_outs=1, fwd=True)

      inputs = load_ops_library.fp8_quantize(
        inputs,
        fwd_fp8_meta_tensors[1][0],
        fwd_fp8_meta_tensors[1][1],
        fp8_meta_index=0,
        fp8_dtype=fp8_dtype_forward,
      )

      query_kernel = load_ops_library.fp8_quantize(
        query_kernel,
        fwd_fp8_meta_tensors[1][0],
        fwd_fp8_meta_tensors[1][1],
        fp8_meta_index=1,
        fp8_dtype=fp8_dtype_forward,
      )

      query = fp8_matmul(
        inputs,
        fwd_fp8_meta_tensors[1][2],
        0,
        fp8_dtype_forward,
        query_kernel,
        fwd_fp8_meta_tensors[1][2],
        1,
        fp8_dtype_forward,
        self.activation_dtype,
        bias=query_bias,
        output_amax=fwd_fp8_meta_tensors[1][0],
        output_scale=fwd_fp8_meta_tensors[1][1],
        output_index=2,
        output_fp8_dtype=fp8_dtype_forward,
        fp8_out=True,
      )

      key_kernel = load_ops_library.fp8_quantize(
        key_kernel,
        fwd_fp8_meta_tensors[1][0],
        fwd_fp8_meta_tensors[1][1],
        fp8_meta_index=3,
        fp8_dtype=fp8_dtype_forward,
      )

      key = fp8_matmul(
        inputs,
        fwd_fp8_meta_tensors[1][2],
        0,
        fp8_dtype_forward,
        key_kernel,
        fwd_fp8_meta_tensors[1][2],
        3,
        fp8_dtype_forward,
        self.activation_dtype,
        bias=key_bias,
        output_amax=fwd_fp8_meta_tensors[1][0],
        output_scale=fwd_fp8_meta_tensors[1][1],
        output_index=4,
        output_fp8_dtype=fp8_dtype_forward,
        fp8_out=True,
      )

      value_kernel = load_ops_library.fp8_quantize(
        value_kernel,
        fwd_fp8_meta_tensors[1][0],
        fwd_fp8_meta_tensors[1][1],
        fp8_meta_index=5,
        fp8_dtype=fp8_dtype_forward,
      )

      value = fp8_matmul(
        inputs,
        fwd_fp8_meta_tensors[1][2],
        0,
        fp8_dtype_forward,
        value_kernel,
        fwd_fp8_meta_tensors[1][2],
        5,
        fp8_dtype_forward,
        self.activation_dtype,
        bias=value_bias,
        output_amax=fwd_fp8_meta_tensors[1][0],
        output_scale=fwd_fp8_meta_tensors[1][1],
        output_index=6,
        output_fp8_dtype=fp8_dtype_forward,
        fp8_out=True,
      )

      query = tf.reshape(
        query, [self.batch, self.seq_len, self.num_attention_heads, self.head_size])
      key = tf.reshape(
        key, [self.batch, self.seq_len, self.num_attention_heads, self.head_size])
      value = tf.reshape(
        value, [self.batch, self.seq_len, self.num_attention_heads, self.head_size])

      query = tf.transpose(query, [0, 2, 1, 3])
      key = tf.transpose(key, [0, 2, 1, 3])
      value = tf.transpose(value, [0, 2, 1, 3])

      context, softmax, attn = fp8_scaled_dot_product_attention(
        query,
        key,
        value,
        qk_scale,
        attention_mask,
        dropout_mask,
        q_scale_inv=fwd_fp8_meta_tensors[1][2],
        k_scale_inv=fwd_fp8_meta_tensors[1][2],
        v_scale_inv=fwd_fp8_meta_tensors[1][2],
        attn_amax=fwd_fp8_meta_tensors[1][0],
        attn_scale=fwd_fp8_meta_tensors[1][1],
        attn_scale_inv=fwd_fp8_meta_tensors[1][2],
        z_amax=fwd_fp8_meta_tensors[1][0],
        z_scale=fwd_fp8_meta_tensors[1][1],
        fp8_meta_index_q=2,
        fp8_meta_index_k=4,
        fp8_meta_index_v=6,
        fp8_meta_index_attn=7,
        fp8_meta_index_z=8,
        dropout_prob=self.attention_dropout,
        fp8_dtype=fp8_dtype_forward,
      )
      context = tf.reshape(context, [self.batch * self.seq_len, self.hidden_size])

      output_kernel = load_ops_library.fp8_quantize(
        output_kernel,
        fwd_fp8_meta_tensors[1][0],
        fwd_fp8_meta_tensors[1][1],
        fp8_meta_index=9,
        fp8_dtype=fp8_dtype_forward,
      )
      output = fp8_matmul(
        context,
        fwd_fp8_meta_tensors[1][2],
        8,
        fp8_dtype_forward,
        output_kernel,
        fwd_fp8_meta_tensors[1][2],
        9,
        fp8_dtype_forward,
        self.activation_dtype,
        bias=output_bias,
      )

      def grad_fn(upstream):
        self.pre_backward()
        fp8_dtype_backward = get_fp8_dtype(self.fp8_meta["recipe"], fwd=False)
        bwd_fp8_meta_tensors = self.get_fp8_meta_tensors(
          upstream, max_fp8_outs=0, fwd=False)

        grad, output_bias_grad = load_ops_library.fp8_quantize_dbias(
          upstream,
          bwd_fp8_meta_tensors[1][0],
          bwd_fp8_meta_tensors[1][1],
          fp8_meta_index=0,
          fp8_dtype=fp8_dtype_backward,
        )
        grad = tf.reshape(grad, upstream.shape)
        output_bias_grad = tf.reshape(output_bias_grad, output_bias.shape)

        output_kernel_grad = fp8_matmul(
          context,
          fwd_fp8_meta_tensors[1][2],
          8,
          fp8_dtype_forward,
          grad,
          bwd_fp8_meta_tensors[1][2],
          0,
          fp8_dtype_backward,
          self.activation_dtype,
          transpose_a=True,
        )

        grad = fp8_matmul(
          grad,
          bwd_fp8_meta_tensors[1][2],
          0,
          fp8_dtype_backward,
          output_kernel,
          fwd_fp8_meta_tensors[1][2],
          9,
          fp8_dtype_forward,
          self.activation_dtype,
          transpose_b=True,
          output_amax=bwd_fp8_meta_tensors[1][0],
          output_scale=bwd_fp8_meta_tensors[1][1],
          output_index=1,
          output_fp8_dtype=fp8_dtype_backward,
          fp8_out=True,
        )

        dq, dk, dv = fp8_scaled_dot_product_attention_grad(
          grad,
          query,
          key,
          value,
          qk_scale,
          softmax,
          dropout_mask,
          attn,
          dz_scale_inv=bwd_fp8_meta_tensors[1][2],
          attn_scale_inv=fwd_fp8_meta_tensors[1][2],
          q_scale_inv=fwd_fp8_meta_tensors[1][2],
          k_scale_inv=fwd_fp8_meta_tensors[1][2],
          v_scale_inv=fwd_fp8_meta_tensors[1][2],
          dp_amax=bwd_fp8_meta_tensors[1][0],
          dp_scale=bwd_fp8_meta_tensors[1][1],
          dp_scale_inv=bwd_fp8_meta_tensors[1][2],
          fp8_meta_index_dz=1,
          fp8_meta_index_attn=7,
          fp8_meta_index_q=2,
          fp8_meta_index_k=4,
          fp8_meta_index_v=6,
          fp8_meta_index_dp=2,
          dropout_prob=self.attention_dropout,
          fp8_dtype_forward=fp8_dtype_forward,
          fp8_dtype_backward=fp8_dtype_backward,
        )

        dq = tf.reshape(dq, query.shape)
        dk = tf.reshape(dk, key.shape)
        dv = tf.reshape(dv, value.shape)

        dq = tf.transpose(dq, [0, 2, 1, 3])
        dk = tf.transpose(dk, [0, 2, 1, 3])
        dv = tf.transpose(dv, [0, 2, 1, 3])

        dq = tf.reshape(dq, [self.batch * self.seq_len, self.hidden_size])
        dk = tf.reshape(dk, [self.batch * self.seq_len, self.hidden_size])
        dv = tf.reshape(dv, [self.batch * self.seq_len, self.hidden_size])

        dq, query_bias_grad = load_ops_library.fp8_quantize_dbias(
          dq,
          bwd_fp8_meta_tensors[1][0],
          bwd_fp8_meta_tensors[1][1],
          fp8_meta_index=3,
          fp8_dtype=fp8_dtype_backward,
        )
        dq = tf.reshape(dq, [self.batch * self.seq_len, self.hidden_size])
        query_bias_grad = tf.reshape(query_bias_grad, query_bias.shape)

        query_kernel_grad = fp8_matmul(
          inputs,
          fwd_fp8_meta_tensors[1][2],
          0,
          fp8_dtype_forward,
          dq,
          bwd_fp8_meta_tensors[1][2],
          3,
          fp8_dtype_backward,
          self.activation_dtype,
          transpose_a=True,
        )
        
        inputs_partial_grad0 = fp8_matmul(
          dq,
          bwd_fp8_meta_tensors[1][2],
          3,
          fp8_dtype_backward,
          query_kernel,
          fwd_fp8_meta_tensors[1][2],
          1,
          fp8_dtype_forward,
          self.activation_dtype,
          transpose_b=True,
        )

        dk, key_bias_grad = load_ops_library.fp8_quantize_dbias(
          dk,
          bwd_fp8_meta_tensors[1][0],
          bwd_fp8_meta_tensors[1][1],
          fp8_meta_index=4,
          fp8_dtype=fp8_dtype_backward,
        )
        dk = tf.reshape(dk, [self.batch * self.seq_len, self.hidden_size])
        key_bias_grad = tf.reshape(key_bias_grad, key_bias.shape)

        key_kernel_grad = fp8_matmul(
          inputs,
          fwd_fp8_meta_tensors[1][2],
          0,
          fp8_dtype_forward,
          dk,
          bwd_fp8_meta_tensors[1][2],
          4,
          fp8_dtype_backward,
          self.activation_dtype,
          transpose_a=True,
        )
        
        inputs_partial_grad1 = fp8_matmul(
          dk,
          bwd_fp8_meta_tensors[1][2],
          4,
          fp8_dtype_backward,
          key_kernel,
          fwd_fp8_meta_tensors[1][2],
          3,
          fp8_dtype_forward,
          self.activation_dtype,
          transpose_b=True,
        )

        dv, value_bias_grad = load_ops_library.fp8_quantize_dbias(
          dv,
          bwd_fp8_meta_tensors[1][0],
          bwd_fp8_meta_tensors[1][1],
          fp8_meta_index=5,
          fp8_dtype=fp8_dtype_backward,
        )
        dv = tf.reshape(dv, [self.batch * self.seq_len, self.hidden_size])
        value_bias_grad = tf.reshape(value_bias_grad, value_bias.shape)

        value_kernel_grad = fp8_matmul(
          inputs,
          fwd_fp8_meta_tensors[1][2],
          0,
          fp8_dtype_forward,
          dv,
          bwd_fp8_meta_tensors[1][2],
          5,
          fp8_dtype_backward,
          self.activation_dtype,
          transpose_a=True,
        )
        
        inputs_partial_grad2 = fp8_matmul(
          dv,
          bwd_fp8_meta_tensors[1][2],
          5,
          fp8_dtype_backward,
          value_kernel,
          fwd_fp8_meta_tensors[1][2],
          5,
          fp8_dtype_forward,
          self.activation_dtype,
          transpose_b=True,
        )

        inputs_grad = (
          inputs_partial_grad0 + inputs_partial_grad1 + inputs_partial_grad2)

        return [
          inputs_grad,
          query_kernel_grad,
          query_bias_grad,
          key_kernel_grad,
          key_bias_grad,
          value_kernel_grad,
          value_bias_grad,
          output_kernel_grad,
          output_bias_grad,
          None,
          None,
          None,
        ]
      
      return output, grad_fn
    
    return fp8_mha_func(
      inputs,
      query_kernel,
      query_bias,
      key_kernel,
      key_bias,
      value_kernel,
      value_bias,
      output_kernel,
      output_bias,
      qk_scale,
      attention_mask,
      dropout_mask,
    )
  
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

  def call(self, inputs, attention_mask, training=None):
    is_training = self._get_training_value(training)
    self.activation_dtype = get_activation_dtype(
      inputs.dtype, self.compute_dtype
    )

    self.fp8_init(num_fp8_inps=0, num_gemms=6, num_fp8_outs=0)

    inputs = cast_if_needed(
      inputs, self.activation_dtype)

    query_kernel = cast_if_needed(
      self.query_kernel, self.activation_dtype)
    query_bias = cast_if_needed(
      self.query_bias, self.activation_dtype)

    key_kernel = cast_if_needed(
      self.key_kernel, self.activation_dtype)
    key_bias = cast_if_needed(
      self.key_bias, self.activation_dtype)

    value_kernel = cast_if_needed(
      self.value_kernel, self.activation_dtype)
    value_bias = cast_if_needed(
      self.value_bias, self.activation_dtype)

    output_kernel = cast_if_needed(
      self.output_kernel, self.activation_dtype)
    output_bias = cast_if_needed(
      self.output_bias, self.activation_dtype)

    attention_mask = cast_if_needed(
      attention_mask, self.activation_dtype
    )
    
    inputs_shape = inputs.shape
    self.seq_len = inputs_shape[1]
    inputs_feature = inputs_shape[-1]
    inputs = tf.reshape(
      inputs, [-1, inputs_feature])
    qk_scale = tf.cast(
      1.0 / tf.math.sqrt(float(self.head_size)),
      dtype=self.activation_dtype)

    if self.fp8:
      assert is_training
      self.batch = inputs_shape[0]
      random_tensor = random_ops.random_uniform(
        shape=[self.batch, self.num_attention_heads,
               self.seq_len, self.seq_len],
        dtype=self.activation_dtype)
      dropout_mask = tf.cast(
        random_tensor >= self.attention_dropout, dtype=self.activation_dtype)
      attention_mask = (1.0 - attention_mask) * -10000.0
      output = self._fp8_mha_forward(
        inputs,
        query_kernel,
        query_bias,
        key_kernel,
        key_bias,
        value_kernel,
        value_bias,
        output_kernel,
        output_bias,
        qk_scale,
        attention_mask,
        dropout_mask,
      )
    else:
      query_layer = tf.matmul(inputs, query_kernel) + query_bias
      key_layer = tf.matmul(inputs, key_kernel) + key_bias
      value_layer = tf.matmul(inputs, value_kernel) + value_bias

      query_layer = tf.reshape(
        query_layer, [-1, self.seq_len, self.num_attention_heads, self.head_size])
      key_layer = tf.reshape(
        key_layer, [-1, self.seq_len, self.num_attention_heads, self.head_size])
      query_layer = tf.transpose(a=query_layer, perm=[0, 2, 1, 3])
      key_layer = tf.transpose(a=key_layer, perm=[0, 2, 1, 3])
      attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
      attention_scores = tf.multiply(attention_scores, qk_scale)
                                     
      attention_mask = tf.expand_dims(attention_mask, axis=[1])
      adder = (1.0 - attention_mask) * -10000.0
      attention_scores += adder

      attention_probs = tf.nn.softmax(attention_scores, axis=3)
      if not (self.attention_dropout is None or self.attention_dropout == 0.0):
        attention_probs = tf.nn.dropout(attention_probs, rate=self.attention_dropout)

      value_layer = tf.reshape(
        value_layer, [-1, self.seq_len, self.num_attention_heads, self.head_size])
      value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])

      context_layer = tf.matmul(attention_probs, value_layer)
      context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])
      context_layer = tf.reshape(
        context_layer,
        [-1, self.num_attention_heads * self.head_size])
      
      output = tf.matmul(context_layer, output_kernel) + output_bias
    
    return output
