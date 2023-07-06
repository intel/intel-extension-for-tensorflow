# Copyright (c) 2022 Intel Corporation
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

def set_config(config=None):
  """set config"""
  if config is None:
    config = config_pb2.ConfigProto()
  if not isinstance(config, config_pb2.ConfigProto):
    raise TypeError('config must be a tf.ConfigProto, but got %s' %
                    type(config))
  config_str = config.SerializeToString()
  ITEX_SetConfig(config_str)

def get_config():
  config_str = ITEX_GetConfig()
  config = config_pb2.ConfigProto()
  config.ParseFromString(config_str)
  return config
