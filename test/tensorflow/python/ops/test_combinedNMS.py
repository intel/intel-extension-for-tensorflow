# Copyright (c) 2022 Intel Corporation
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
from tensorflow.python.ops import image_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
import numpy as np


class CombinedNMSTest(test_util.TensorFlowTestCase):
    # this test is to verify the result of combinedNMS

    def __init__(self, methodName="run combinedNMS test"):
        super().__init__(methodName)
        self.max_output_size_per_class = 100
        self.max_total_size = 100
        self.iou_threshold = 0.6
        self.q_val = 1
        self.num_batches = 1
        self.num_classes = 90
        self.num_boxes = 1917
        self.boxes_np = None
        self.scores_np = None

    def _init_boxes_and_scores(self):
        self.boxes_np = np.random.rand(self.num_batches, self.num_boxes,
                                       self.q_val, 4).astype(np.float32)
        self.scores_np = []
        for bs in range(self.num_batches):
            scores_per_bs = []
            for cl in range(self.num_classes):
                scores = np.random.uniform(low=0.00001,
                                           high=0.45,
                                           size=self.num_boxes).astype(
                                               np.float32)
                scores_per_bs.append(scores)
            self.scores_np.append(scores_per_bs)
        self.scores_np = np.array(self.scores_np).transpose(0, 2, 1)

    def _run(self, dev, threshold=0):
        # wenjie: soft_placement is True by default, so it works on XPU
        with ops.device("/device:%s:0" % dev):
            res = image_ops.combined_non_max_suppression(
                boxes=self.boxes_np,
                scores=self.scores_np,
                max_output_size_per_class=self.max_output_size_per_class,
                max_total_size=self.max_total_size,
                iou_threshold=self.iou_threshold,
                score_threshold=threshold)
        return res

    def testCombinedNMS(self):
        self._init_boxes_and_scores()
        cpu_res = self._run("cpu")
        xpu_res = self._run("xpu")
        for i in range(len(xpu_res)):
            self.assertAllEqual(xpu_res[i], cpu_res[i])

    def testCombinedNMSWithHighThreshold(self):
        self._init_boxes_and_scores()
        cpu_res = self._run("cpu", 0.4)
        xpu_res = self._run("xpu", 0.4)
        for i in range(len(xpu_res)):
            self.assertAllEqual(xpu_res[i], cpu_res[i])


if __name__ == "__main__":
    test.main()
