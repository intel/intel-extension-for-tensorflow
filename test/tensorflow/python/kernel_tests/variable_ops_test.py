"""Tests for variable_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables

_NP_TO_TF = {
    np.float16: dtypes.float16,
    np.float32: dtypes.float32,
    np.int32: dtypes.int32,
    np.int64: dtypes.int64,
}


class VariableOpTest(test.TestCase):

  def _initFetch(self, x, tftype, use_gpu=None):
    with self.test_session(use_gpu=use_gpu):
      p = state_ops.variable_op(x.shape, tftype)
      op = state_ops.assign(p, x)
      op.op.run()
      return self.evaluate(p)

  def _testTypes(self, vals):
    for dtype in [
        np.float16, np.float32,
        np.int32, np.int64
    ]:
      self.setUp()
      x = vals.astype(dtype)
      tftype = _NP_TO_TF[dtype]
      self.assertAllEqual(x, self._initFetch(x, tftype, use_gpu=False))
      # NOTE(touts): the GPU test should pass for all types, whether the
      # Variable op has an implementation for that type on GPU as we expect
      # that Variable and Assign have GPU implementations for matching tf.
      self.assertAllEqual(x, self._initFetch(x, tftype, use_gpu=True))

  @test_util.run_deprecated_v1
  def testTemporaryVariable(self):
    with test_util.use_gpu():
      var = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="foo")
      var = state_ops.assign(var, [[4.0, 5.0]])
      var = state_ops.assign_add(var, [[6.0, 7.0]])
      final = gen_state_ops.destroy_temporary_variable(var, var_name="foo")
      self.assertAllClose([[10.0, 12.0]], self.evaluate(final))

  @test_util.run_deprecated_v1
  def testDestroyNonexistentTemporaryVariable(self):
    with test_util.use_gpu():
      var = gen_state_ops.temporary_variable([1, 2], dtypes.float32)
      final = gen_state_ops.destroy_temporary_variable(var, var_name="bad")
      with self.assertRaises(errors.NotFoundError):
        self.evaluate(final)

  @test_util.run_deprecated_v1
  def testDuplicateTemporaryVariable(self):
    with test_util.use_gpu():
      var1 = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="dup")
      var1 = state_ops.assign(var1, [[1.0, 2.0]])
      var2 = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="dup")
      var2 = state_ops.assign(var2, [[3.0, 4.0]])
      final = var1 + var2
      with self.assertRaises(errors.AlreadyExistsError):
        self.evaluate(final)

  @test_util.run_deprecated_v1
  def testDestroyTemporaryVariableTwice(self):
    with test_util.use_gpu():
      var = gen_state_ops.temporary_variable([1, 2], dtypes.float32)
      val1 = gen_state_ops.destroy_temporary_variable(var, var_name="dup")
      val2 = gen_state_ops.destroy_temporary_variable(var, var_name="dup")
      final = val1 + val2
      with self.assertRaises(errors.NotFoundError):
        self.evaluate(final)

  @test_util.run_deprecated_v1
  def testTemporaryVariableNoLeak(self):
    with test_util.use_gpu():
      var = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="bar")
      final = array_ops.identity(var)
      self.evaluate(final)

  @test_util.run_deprecated_v1
  def testTwoTemporaryVariablesNoLeaks(self):
    with test_util.use_gpu():
      var1 = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="var1")
      var2 = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="var2")
      final = var1 + var2
      self.evaluate(final)

  @test_util.run_deprecated_v1
  def testAssignDependencyAcrossDevices(self):
    with test_util.use_gpu():
      # The variable and an op to increment it are on the GPU.
      var = state_ops.variable_op([1], dtypes.float32)
      self.evaluate(state_ops.assign(var, [1.0]))
      increment = state_ops.assign_add(var, [1.0])
      with ops.control_dependencies([increment]):
        with test_util.force_cpu():
          # This mul op is pinned to the CPU, but reads the variable from the
          # GPU. The test ensures that the dependency on 'increment' is still
          # honored, i.e., the Send and Recv from GPU to CPU should take place
          # only after the increment.
          result = math_ops.multiply(var, var)
      self.assertAllClose([4.0], self.evaluate(result))

  # Currently, we don't have Assign GPU kernel, because of lacking C-API. However, 
  # this UT force place Assign op on GPU, so we disable it currently,
  # @test_util.run_deprecated_v1
  # def testIsVariableInitialized(self):
  #   for use_gpu in [True, False]:
  #     with self.test_session(use_gpu=use_gpu):
  #       v0 = state_ops.variable_op([1, 2], dtypes.float32)
  #       self.assertEqual(False, variables.is_variable_initialized(v0).eval())
  #       state_ops.assign(v0, [[2.0, 3.0]]).eval()
  #       self.assertEqual(True, variables.is_variable_initialized(v0).eval())


if __name__ == "__main__":
  test.main()
