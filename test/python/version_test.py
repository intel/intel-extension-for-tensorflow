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


#!/usr/bin/env python
# coding=utf-8
from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow import test
from intel_extension_for_tensorflow.python.version import __version__
from intel_extension_for_tensorflow.python.version import VERSION
from intel_extension_for_tensorflow.python.version import COMPILER_VERSION
from intel_extension_for_tensorflow.python.version import GIT_VERSION
from intel_extension_for_tensorflow.python.version import TF_COMPATIBLE_VERSION
from intel_extension_for_tensorflow.python.version import ONEDNN_GIT_VERSION
 

class VersionTest(test_util.TensorFlowTestCase):
    """test version info."""

    def testVersionType(self):
        self.assertEqual(type(__version__), str)
        self.assertEqual(type(VERSION), str)
        self.assertEqual(type(COMPILER_VERSION), str)
        self.assertEqual(type(GIT_VERSION), str)
        self.assertEqual(type(TF_COMPATIBLE_VERSION), str)
        self.assertEqual(type(ONEDNN_GIT_VERSION), str)

    def testVersion(self):
        self.assertEqual(__version__, VERSION)
        # 1.1.1
        self.assertRegex(__version__, r'([0-9].){2}[0-9]')
        # v1.1.1-abcd1234
        self.assertRegex(GIT_VERSION, r'v([0-9]+.){2}[0-9]+-[0-9a-z]{8}')
        # gcc-1.1.1, dpcpp-a.0.1
        self.assertRegex(COMPILER_VERSION, r'gcc-([0-9]+.){2}[0-9]+, dpcpp-([0-9a-z].)+[0-9]+')
        # v1.1.1-abcd1234
        self.assertRegex(ONEDNN_GIT_VERSION, r'v([0-9]+.){2}[0-9]+-[0-9a-z]{8}')
        self.assertNotEmpty(TF_COMPATIBLE_VERSION)

if __name__ == '__main__':
    test.main()
