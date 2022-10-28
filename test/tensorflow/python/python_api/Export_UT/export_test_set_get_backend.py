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



import os
import tensorflow as tf
import intel_extension_for_tensorflow as itex
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

backend_gpuflag = 2.0
GPU_BACKEND = 'GPU'

class SetGetBackendTest(test_util.TensorFlowTestCase):
    """test set_backend and get_backend itex python api"""

    @test_util.run_deprecated_v1
    def testSetGetBackend_gpu(self):
        os.environ['ITEX_XPU_BACKEND'] = GPU_BACKEND
        current_backend = 0.0
        if(itex.get_backend() == GPU_BACKEND.encode('utf-8')):
            current_backend = backend_gpuflag
        else:
            current_backend = 0.0
        self.assertAllClose(current_backend, backend_gpuflag, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test.main()
