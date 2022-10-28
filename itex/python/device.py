# Copyright (c) 2021 Intel Corporation
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

"""device"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from intel_extension_for_tensorflow.python._pywrap_itex import *
from intel_extension_for_tensorflow.core.utils.protobuf import config_pb2

_VALID_DEVICE_BACKENDS = frozenset({"CPU", "GPU", "AUTO"})


def set_backend(backend, config=None):
  """set backend"""
  if config is None:
    config = config_pb2.ConfigProto()
  if not isinstance(config, config_pb2.ConfigProto):
    raise TypeError('config must be a tf.ConfigProto, but got %s' %
                    type(config))
  if backend.upper() in _VALID_DEVICE_BACKENDS:
    config_str = config.SerializeToString()
    ITEX_SetBackend(backend.upper(), config_str)
  else:
    raise ValueError("Cannot specify %s as XPU backend, Only %s is VALID"
                     % (backend, _VALID_DEVICE_BACKENDS))


def get_backend():
  return ITEX_GetBackend()
