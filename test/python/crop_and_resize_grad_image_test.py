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

import tensorflow as tf

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_image_ops

BATCH_SIZE = 1
NUM_BOXES = 5
CHANNELS = 3
CROP_SIZE = (24, 24)

IMAGE_DTYPE = [dtypes.float32]

class CropAndResizeGradImageTest(test_util.TensorFlowTestCase):
  """test CropAndResizeGradImage op"""

  def _test_impl(self, in_size, dtype):
    grads = tf.random.normal(
        shape=(NUM_BOXES, CROP_SIZE[0], CROP_SIZE[1], CHANNELS), dtype=dtypes.float32)
    boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
    box_indices = tf.random.uniform(
        shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=dtypes.int32)
    image_size = constant_op.constant(
        [BATCH_SIZE, in_size[0], in_size[1], CHANNELS], dtype=dtypes.int32)
    out_gpu = gen_image_ops.crop_and_resize_grad_image(grads, boxes, box_indices, image_size, dtype)
    self.assertEqual(out_gpu.get_shape().as_list(), image_size.numpy().tolist())
    self.assertEqual(out_gpu.dtype, dtype)
    self.evaluate(out_gpu)
    
  def testCropAndResizeGradImage(self):
    for dtype in IMAGE_DTYPE:
        for in_size in [[256, 256], [480, 800], [240, 320]]:
            self._test_impl(in_size, dtype)
    
if __name__ == "__main__":
  test.main()
