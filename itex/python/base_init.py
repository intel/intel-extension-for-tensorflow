# Copyright (c) 2021-2022 Intel Corporation
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
'''Init file for graph optimizer config, custom ops'''

import tensorflow  # pylint: disable=unused-import
import intel_extension_for_tensorflow_lib  # pylint: disable=unused-import
from intel_extension_for_tensorflow.python.config import set_config  # pylint: disable=unused-import
from intel_extension_for_tensorflow.python.config import get_config  # pylint: disable=unused-import
from intel_extension_for_tensorflow.python.device import get_backend  # pylint: disable=unused-import
from intel_extension_for_tensorflow.python.device import is_xehpc  # pylint: disable=unused-import
from intel_extension_for_tensorflow.python import ops  # pylint: disable=unused-import,line-too-long
from intel_extension_for_tensorflow.python.version import __version__  # pylint: disable=unused-import
from intel_extension_for_tensorflow.python import version  # pylint: disable=unused-import
from intel_extension_for_tensorflow.python import test_func  # pylint: disable=unused-import

from intel_extension_for_tensorflow.core.utils.protobuf.config_pb2 import *  # pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from intel_extension_for_tensorflow.python.experimental_ops_override import experimental_ops_override
