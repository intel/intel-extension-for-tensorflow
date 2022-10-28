import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32] 
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32]  # BF16 is not supported by CUDA
    
ITERATION = 5

box_size_list = [[12288,4], [196608,4], [3072,4], [49152,4], [768,4]]
score_size_list = [[12288], [196608], [3072], [49152], [768]]

class NonMaxSuppressionV2Test(test.TestCase):
    def _test_impl(self, box_size, score_size, dtype):
        np.random.seed(4)
        boxes_np = np.random.random(size=box_size)
        scores_np = np.random.random(size=score_size)
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(3)
        iou_threshold = constant_op.constant(0.5, dtype=dtype)  
        flush_cache()
        selected_indices = gen_image_ops.non_max_suppression_v2(
            boxes, scores, max_output_size, iou_threshold)

    @add_profiling
    @multi_run(ITERATION)
    def testOp(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for index in range(len(box_size_list)):
                self._test_impl(box_size_list[index], score_size_list[index], dtype)
   
                
class NonMaxSuppressionV3Test(test.TestCase):
    def _test_impl(self, box_size, score_size, dtype):
        np.random.seed(4)
        boxes_np = np.random.random(size=box_size)
        scores_np = np.random.random(size=score_size)
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(3)
        iou_threshold = constant_op.constant(0.5, dtype=dtype)  
        score_threshold_np = float("-inf")
        score_threshold = constant_op.constant(score_threshold_np, dtype=dtype)
        flush_cache()
        selected_indices = gen_image_ops.non_max_suppression_v3(
            boxes, scores, max_output_size, iou_threshold, score_threshold)

    @add_profiling
    @multi_run(ITERATION)
    def testOp(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for index in range(len(box_size_list)):
                self._test_impl(box_size_list[index], score_size_list[index], dtype)


class NonMaxSuppressionV4Test(test.TestCase):
    def _test_impl(self, box_size, score_size, dtype):
        np.random.seed(4)
        boxes_np = np.random.random(size=box_size)
        scores_np = np.random.random(size=score_size)
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(3)
        iou_threshold = constant_op.constant(0.5, dtype=dtype)  
        score_threshold_np = float("-inf")
        score_threshold = constant_op.constant(score_threshold_np, dtype=dtype)
        flush_cache()
        selected_indices, _ = gen_image_ops.non_max_suppression_v4(
            boxes, scores, max_output_size, iou_threshold, score_threshold)

    @add_profiling
    @multi_run(ITERATION)
    def testOp(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for index in range(len(box_size_list)):
                self._test_impl(box_size_list[index], score_size_list[index], dtype)


if __name__ == '__main__':
    test.main()    