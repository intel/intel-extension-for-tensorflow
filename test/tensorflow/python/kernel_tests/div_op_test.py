"""Tests for the raw div operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

integer_dtypes = [np.uint16, np.int32, np.int64]
float_dtypes = [np.float32, np.float64]
complex_dtypes = [np.complex64, np.complex128]

class DivTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testDiv(self):
    
    for dtype in integer_dtypes:
      x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(dtype)
      y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(dtype)
      self._compareBoth(x, y)

    for dtype in float_dtypes:
      x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(dtype)
      y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(dtype)
      self._compareBoth(x, y)
    for dtype in complex_dtypes:
      x = complex(1, 1) * np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(
        dtype)
      y = complex(1, 1) * np.linspace(20, -20, 6).reshape(1, 3, 2).astype(
        dtype)
      self._compareBoth(x, y)

  def _compareBoth(self, x, y):
    np_ans = np.divide(x, y).astype(x.dtype)
    for use_gpu in (False, True):
      with self.session(use_gpu=use_gpu):
        inx = ops.convert_to_tensor(x)
        iny = ops.convert_to_tensor(y)
        tf_out = tf.raw_ops.Div(x=inx, y=iny)
        self.assertAllCloseAccordingToType(np_ans, tf_out)

if __name__ == "__main__":
  test.main()
