"""Functional tests for quantized operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
import intel_extension_for_tensorflow as itex

import numpy as np
import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

import tensorflow as tf

os.environ["ITEX_LAYOUT_OPT"] = "0"
os.environ["ITEX_NATIVE_FORMAT"] = "1"

class QuantizedConvOldAPI(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedConvOldAPI, self).__init__(method_name)

  # Quantize + Conv2D + Relu + Requantize u8s8 test
  @test_util.run_deprecated_v1
  def testQuantizedConv2DAndReluAndRequantize(self):
    # ITEX does not write QuantizedConv2DAndReluAndRequantize on CPU
    # so that this op would run into official tensorflow's implementation, which have difference input arguments.
    if not test.is_gpu_available():
      self.skipTest("Skip on CPU")

    with ops.name_scope("test"):
      x_f32_np = np.random.uniform(low=-0.0, high=5.0, size=(1, 6, 6, 4)).astype(np.float32)
      x_f32 = constant_op.constant(x_f32_np)
      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.quint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)
      
      y_f32_np = np.random.uniform(low=-2.0, high=2.0, size=(3, 3, 4, 4)).astype(np.float32)
      y_f32 = constant_op.constant(y_f32_np)
      y_min = tf.math.reduce_min(y_f32, axis=(0, 1, 2))
      y_max = tf.math.reduce_max(y_f32, axis=(0, 1, 2))
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True, axis=3)

      conv_f32 = tf.nn.relu(tf.nn.conv2d(x_f32, y_f32, [1,1,1,1], padding="SAME"))
      z_freezed_min = tf.math.reduce_min(conv_f32)
      z_freezed_max = tf.math.reduce_max(conv_f32)

      conv_int8_req, conv_freezed_min, conv_freezed_max = load_ops_library.QuantizedConv2DAndReluAndRequantize(input=x_int8, filter=y_int8, 
                                                                               min_input=x_min, max_input=x_max, min_filter=y_min, max_filter=y_max, 
                                                                               min_freezed_output=z_freezed_min, max_freezed_output=z_freezed_max,
                                                                               strides=[1, 1, 1, 1], padding="SAME", out_type=dtypes.quint8) 
      conv_int8 = array_ops.dequantize(conv_int8_req, conv_freezed_min, conv_freezed_max, mode="SCALED", narrow_range=True)

      conv_f32 = array_ops.identity(conv_f32)
      conv_int8 = array_ops.identity(conv_int8)
      
      conv_int8_res = self.evaluate(conv_int8)
      conv_f32_res = self.evaluate(conv_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(conv_int8_res, conv_f32_res, rtol=0.3, atol=0.3)

  #  conv + bias + requantize s8s8s8 test
  @test_util.run_deprecated_v1
  def testConvBias(self):
    with ops.name_scope("test"):
      x_f32_np = np.random.uniform(low=-5.0, high=5.0, size=(1, 6, 6, 4)).astype(np.float32)
      x_f32 = constant_op.constant(x_f32_np)

      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)
      y_f32_np = np.random.uniform(low=-2.0, high=2.0, size=(3, 3, 4, 4)).astype(np.float32)
      y_f32 = constant_op.constant(y_f32_np)

      y_min = tf.math.reduce_min(y_f32, axis=(0, 1, 2))
      y_max = tf.math.reduce_max(y_f32, axis=(0, 1, 2))
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True, axis=3)

      bias_f32_np = np.random.uniform(low=-1, high=1.0, size=(4)).astype(np.float32)
      bias_f32 = constant_op.constant(bias_f32_np)

      conv_f32 = tf.nn.bias_add(tf.nn.conv2d(x_f32, y_f32, [1,1,1,1], padding="SAME"), bias_f32)

      z_freezed_min = tf.math.reduce_min(conv_f32)
      z_freezed_max = tf.math.reduce_max(conv_f32)

      conv_int8_req, conv_freezed_min, conv_freezed_max = load_ops_library.QuantizedConv2DWithBiasAndRequantize(input=x_int8, filter=y_int8, bias=bias_f32, 
                                                                               min_input=x_min, max_input=x_max, min_filter=y_min, max_filter=y_max, 
                                                                               min_freezed_output=z_freezed_min, max_freezed_output=z_freezed_max,
                                                                               strides=[1, 1, 1, 1], padding="SAME",out_type=dtypes.qint8) 
      conv_int8 = array_ops.dequantize(conv_int8_req, conv_freezed_min, conv_freezed_max, mode="SCALED", narrow_range=True)

      conv_f32 = array_ops.identity(conv_f32)
      conv_int8 = array_ops.identity(conv_int8)
      
      conv_int8_res = self.evaluate(conv_int8)
      conv_f32_res = self.evaluate(conv_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(conv_int8_res, conv_f32_res, rtol=0.3, atol=0.3)

  #  conv + bias + relu + requantize u8s8u8 test
  @test_util.run_deprecated_v1
  def testConvBiasRelu(self):
    with ops.name_scope("test"):
      x_f32_np = np.random.uniform(low=-0.0, high=5.0, size=(1, 6, 6, 4)).astype(np.float32)
      x_f32 = constant_op.constant(x_f32_np)

      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.quint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)
      y_f32_np = np.random.uniform(low=-2.0, high=2.0, size=(3, 3, 4, 4)).astype(np.float32)
      y_f32 = constant_op.constant(y_f32_np)

      y_min = tf.math.reduce_min(y_f32, axis=(0, 1, 2))
      y_max = tf.math.reduce_max(y_f32, axis=(0, 1, 2))
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True, axis=3)

      bias_f32_np = np.random.uniform(low=-1, high=1.0, size=(4)).astype(np.float32)
      bias_f32 = constant_op.constant(bias_f32_np)

      conv_f32 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_f32, y_f32, [1,1,1,1], padding="SAME"), bias_f32))

      z_freezed_min = tf.math.reduce_min(conv_f32)
      z_freezed_max = tf.math.reduce_max(conv_f32)

      conv_int8_req, conv_freezed_min, conv_freezed_max = load_ops_library.QuantizedConv2DWithBiasAndReluAndRequantize(input=x_int8, filter=y_int8, bias=bias_f32, 
                                                                               min_input=x_min, max_input=x_max, min_filter=y_min, max_filter=y_max, 
                                                                               min_freezed_output=z_freezed_min, max_freezed_output=z_freezed_max,
                                                                               strides=[1, 1, 1, 1], padding="SAME", out_type=dtypes.quint8) 
      conv_int8 = array_ops.dequantize(conv_int8_req, conv_freezed_min, conv_freezed_max, mode="SCALED", narrow_range=True)

      conv_f32 = array_ops.identity(conv_f32)
      conv_int8 = array_ops.identity(conv_int8)
      
      conv_int8_res = self.evaluate(conv_int8)
      conv_f32_res = self.evaluate(conv_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(conv_int8_res, conv_f32_res, rtol=0.3, atol=0.3)

  #  conv + bias + add + relu + requantize u8s8u8 + u8 test
  @test_util.run_deprecated_v1
  def testConvBiasSumRelu(self):
    with ops.name_scope("test"):
      x_f32_np = np.random.uniform(low=-0.0, high=5.0, size=(1, 6, 6, 4)).astype(np.float32)
      x_f32 = constant_op.constant(x_f32_np)

      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.quint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      y_f32_np = np.random.uniform(low=-2.0, high=2.0, size=(3, 3, 4, 4)).astype(np.float32)
      y_f32 = constant_op.constant(y_f32_np)

      y_min = tf.math.reduce_min(y_f32, axis=(0, 1, 2))
      y_max = tf.math.reduce_max(y_f32, axis=(0, 1, 2))
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True, axis=3)

      bias_f32_np = np.random.uniform(low=-1, high=1.0, size=(4)).astype(np.float32)
      bias_f32 = constant_op.constant(bias_f32_np)

      a_f32_np = np.random.uniform(low=-0.0, high=5.0, size=(1, 6, 6, 4)).astype(np.float32)
      a_f32 = constant_op.constant(a_f32_np)

      a_min = tf.math.reduce_min(a_f32)
      a_max = tf.math.reduce_max(a_f32)
      a_int8, a_min, a_max = array_ops.quantize(a_f32, a_min, a_max, T=dtypes.quint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      conv_f32 = tf.nn.relu(tf.math.add(tf.nn.bias_add(tf.nn.conv2d(x_f32, y_f32, [1,1,1,1], padding="SAME"), bias_f32), a_f32))

      z_freezed_min = tf.math.reduce_min(conv_f32)
      z_freezed_max = tf.math.reduce_max(conv_f32)

      conv_int8_req, conv_freezed_min, conv_freezed_max = load_ops_library.QuantizedConv2DWithBiasSumAndReluAndRequantize(input=x_int8, filter=y_int8, bias=bias_f32, 
                                                                               min_input=x_min, max_input=x_max, min_filter=y_min, max_filter=y_max, 
                                                                               min_freezed_output=z_freezed_min, max_freezed_output=z_freezed_max,
                                                                               summand=a_int8, min_summand=a_min, max_summand=a_max, 
                                                                               strides=[1, 1, 1, 1], padding="SAME", out_type=dtypes.quint8) 
      conv_int8 = array_ops.dequantize(conv_int8_req, conv_freezed_min, conv_freezed_max, mode="SCALED", narrow_range=True)

      conv_f32 = array_ops.identity(conv_f32)
      conv_int8 = array_ops.identity(conv_int8)
      
      conv_int8_res = self.evaluate(conv_int8)
      conv_f32_res = self.evaluate(conv_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(conv_int8_res, conv_f32_res, rtol=0.3, atol=0.3)


# Enable the Conv INT8 new API, once we figure out way to set Tbias, Tinput via python wrapper
class QuantizedConvNewAPI(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedConvNewAPI, self).__init__(method_name)

  #  conv + bias + dequantize s8s8f32 test
  @test_util.run_deprecated_v1
  def testConvBias(self):

    with ops.name_scope("test"):
      x_f32_np = np.random.uniform(low=-5.0, high=5.0, size=(1, 6, 6, 4)).astype(np.float32)
      x_f32 = constant_op.constant(x_f32_np)

      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)
      y_f32_np = np.random.uniform(low=-2.0, high=2.0, size=(3, 3, 4, 4)).astype(np.float32)
      y_f32 = constant_op.constant(y_f32_np)

      y_min = tf.math.reduce_min(y_f32, axis=(0, 1, 2))
      y_max = tf.math.reduce_max(y_f32, axis=(0, 1, 2))
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True, axis=3)

      bias_f32_np = np.random.uniform(low=-1, high=1.0, size=(4)).astype(np.float32)
      bias_f32 = constant_op.constant(bias_f32_np)

      conv_f32 = tf.nn.bias_add(tf.nn.conv2d(x_f32, y_f32, [1,1,1,1], padding="SAME"), bias_f32)

      z_freezed_min = tf.math.reduce_min(conv_f32)
      z_freezed_max = tf.math.reduce_max(conv_f32)

      conv_int8_bf16, _ = load_ops_library._QuantizedConv2D(device_inputs=[x_int8, y_int8, bias_f32], host_inputs=[x_min, x_max, y_min, y_max, 
                                                    z_freezed_min, z_freezed_max], Tdevice_outputs=[dtypes.bfloat16], Thost_outputs=[], 
                                                    dilations = [1, 1, 1, 1], 
                                                    fused_ops = [b'BiasAdd', b'Dequantize'], out_type=dtypes.bfloat16,
                                                    Tinput=dtypes.qint8, Tfilter=dtypes.qint8, 
                                                    Tbias=dtypes.float32, Tsummand=dtypes.bfloat16,
                                                    strides=[1, 1, 1, 1], padding="SAME") 
      
      conv_int8 = math_ops.cast(conv_int8_bf16, dtypes.float32)
      
      conv_f32 = array_ops.identity(conv_f32)
      conv_int8 = array_ops.identity(conv_int8[0])
      
      conv_int8_res = self.evaluate(conv_int8)
      conv_f32_res = self.evaluate(conv_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(conv_int8_res, conv_f32_res, rtol=0.3, atol=0.3)


if __name__ == "__main__":
  test.main()
