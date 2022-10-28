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


import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test

    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]
except ImportError:
    from tensorflow.python.platform import test

    FLOAT_COMPUTE_TYPE = [
        dtypes.float32,
        dtypes.float16,
    ]  # BF16 is not supported by CUDA

ITERATION = 5
BATCH_SIZE = 1
NUM_BOXES = 5
CHANNELS = 3
CROP_SIZE = (24, 24)


class CropAndResizeGradImageTest(test.TestCase):
    def _test_impl(self, in_size, dtype):
        IMAGE_HEIGHT = in_size[0]
        IMAGE_WIDTH = in_size[1]
        grads = tf.random.normal(
            shape=(NUM_BOXES, CROP_SIZE[0], CROP_SIZE[1], CHANNELS)
        )
        boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
        box_indices = tf.random.uniform(
            shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32
        )
        flush_cache()
        out_gpu = tf.raw_ops.CropAndResizeGradImage(
            grads=grads,
            boxes=boxes,
            box_ind=box_indices,
            image_size=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
            T = dtype
        )

    @add_profiling
    @multi_run(ITERATION)
    def testCropAndResizeGradImage(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in [[256, 256], [480, 800], [240, 320]]:
                self._test_impl(in_size, dtype)


if __name__ == "__main__":
    test.main()
