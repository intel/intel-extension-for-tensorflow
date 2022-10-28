"""Tests for Erfinv Op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test

import importlib

import numpy as np

from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging

def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")

class ErfInvTest(test.TestCase):

  def testErfInvValues(self):
    with self.cached_session():
      if not special:
        return

      x = np.linspace(0., 1.0, 50).astype(np.float32)

      expected_x = special.erfinv(x)
      x = math_ops.erfinv(x)
      self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

  def testErfInvValuesDouble(self):
    with self.cached_session():
      if not special:
        return

      x = np.linspace(0., 1.0, 50).astype(np.float64)

      expected_x = special.erfinv(x)
      x = math_ops.erfinv(x)
      self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

if __name__ == "__main__":
  test.main()
