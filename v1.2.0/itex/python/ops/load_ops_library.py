# Copyright (c) 2021-2022 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=missing-module-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import imp
import hashlib

from tensorflow.python.framework import _pywrap_python_op_gen
from tensorflow.python.client import pywrap_tf_session as py_tf

def _load_ops_library():
  """Loads a TensorFlow plugin, containing custom ops and kernels.

  Pass "library_filename" to a platform-specific mechanism for dynamically
  loading a library. The rules for determining the exact location of the
  library are platform-specific and are not documented here. When the
  library is loaded, ops and kernels registered in the library via the
  `REGISTER_*` macros are made available in the TensorFlow process. Note
  that ops with the same name as an existing op are rejected and not
  registered with the process.

  Args:
    library_filename: Path to the plugin.
      Relative or absolute filesystem path to a dynamic library file.

  Returns:
    A python module containing the Python wrappers for Ops defined in
    the plugin.

  Raises:
    RuntimeError: when unable to load the library or get the python wrappers.
  """
  buf = py_tf.TF_GetAllOpList()
  try:
    wrappers = _pywrap_python_op_gen.GetPythonWrappers(  # pylint: disable=c-extension-no-member
        py_tf.TF_GetBuffer(buf))
  finally:
    # Delete the buf to release any memory held in C
    # that are no longer needed.
    py_tf.TF_DeleteBuffer(buf)

  # Get a unique name for the module.
  module_name = hashlib.sha1(wrappers).hexdigest()
  if module_name in sys.modules:
    return sys.modules[module_name]
  module = imp.new_module(module_name)
  # pylint: disable=exec-used
  exec(wrappers, module.__dict__)
  # Allow this to be recognized by AutoGraph.
  setattr(module, '_IS_TENSORFLOW_PLUGIN', True)
  sys.modules[module_name] = module
  return module

load_ops_library = _load_ops_library()
