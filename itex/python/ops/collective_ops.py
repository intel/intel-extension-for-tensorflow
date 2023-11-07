# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ops for XPU collective operations."""

import threading

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops

_module_lock = threading.Lock()
_shared_name_counter = 0

def _get_shared_name():
  global _shared_name_counter

  with _module_lock:
    val = _shared_name_counter
    _shared_name_counter += 1
  return 'c%s' % val

def _check_device(tensor, expected=None):
  if not device.canonical_name(tensor.device):
    raise ValueError(f'Device assignment for tensor={tensor} required for ITEX '
                     'collective ops')
  if expected and expected != tensor.device:
    raise ValueError(f'Expected device {expected}, got {tensor.device} for '
                     f'tensor={tensor}.')

def all_sum(tensors):
  """Returns a list of tensors with the all-reduce sum across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to sum; must be assigned
      to XPU devices.

  Returns:
    List of tensors, each with the sum of the input tensors, where tensor i has
    the same device as `tensors[i]`.
  """
  return _apply_all_reduce('sum', tensors)

def _apply_all_reduce(reduction, tensors):
  """Helper function."""
  if not tensors:
    raise ValueError('Must pass >0 tensors to all reduce operations')

  shared_name = _get_shared_name()

  def _all_reduce():
    """Call allreduce."""
    res = []
    for t in tensors:
      _check_device(t)
      with ops.device(t.device):
        res.append(
            load_ops_library.itex_all_reduce_send(
                input=t,
                reduction=reduction,
                num_devices=len(tensors),
                shared_name=shared_name))
    return res

  if context.executing_eagerly():
    return def_function.function(_all_reduce)()
  else:
    return _all_reduce()
