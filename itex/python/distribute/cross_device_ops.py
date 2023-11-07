# Copyright (c) 2023 Intel Corporation
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

from intel_extension_for_tensorflow.python.ops import collective_ops
from tensorflow.python.distribute import cross_device_ops
from tensorflow.python.platform import tf_logging as logging

def aggregate_gradients(replica_grads):
  """Aggregate gradients using allreduce."""
  agg_all_g_and_v = []
  for single_g_and_v in zip(*replica_grads):
    single_grads = [g for g, _ in single_g_and_v]
    agg_grads = collective_ops.all_sum(single_grads)
    agg_all_g_and_v.append(
        [(g, v) for g, (_, v) in zip(agg_grads, single_g_and_v)])

  agg_all_g_and_v = list(zip(*agg_all_g_and_v))

  return agg_all_g_and_v

class ItexAllReduce(cross_device_ops.AllReduceCrossDeviceOps):
  """ITEX all-reduce implementation of CrossDeviceOps.

  For the batch API, tensors will be repacked or aggregated 
  for more efficient cross-device transportation.

  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.

  Here is how you can use `ItexAllReduce` in `tf.distribute.MirroredStrategy`:


  ```
    gpus = tf.config.list_logical_devices('XPU')
    strategy = tf.distribute.MirroredStrategy(
      devices=gpus,
      cross_device_ops=itex.distribute.ItexAllReduce())
  ```
  """

  def __init__(self, num_packs=1):
    """Initializes the object.

    Args:
      num_packs: a non-negative integer. The number of packs to split values
        into. If zero, no packing will be done.

    Raises:
      ValueError: if `num_packs` is negative.
    """
    if num_packs < 0:
      raise ValueError(
          "ItexAllreduce requires num_packs >= 0, but {} is specified".format(
              num_packs))
    super(ItexAllReduce, self).__init__(num_packs=num_packs)

    # Add CollectiveOps' OpName into ASYNC_STATEFUL_OPS to avoid auto_control_deps 
    # adding control edges between CollectiveOps of different devices, which would
    # cause dead lock.
    import tensorflow.python.framework.auto_control_deps as auto_control_deps
    auto_control_deps.ASYNC_STATEFUL_OPS = set(auto_control_deps.ASYNC_STATEFUL_OPS)
    auto_control_deps.ASYNC_STATEFUL_OPS.add("ItexAllReduceSend")

  def _do_batch_all_reduce(self, reduce_op, dense_values):
    """Run batch all-reduces.
    
    Both all-reduces and batch all-reduces use this api.

    This api only works for dense tensor. Sparse tensors will fallback to 
    `tf.distribute.ReductionToOneDevice` by base class, which is aligned with
    NcclAllReduce.
    """
    logging.log_first_n(
        logging.INFO,
        "ItexAllReduce: %d all-reduces, num_packs = %d" %
        (len(dense_values), self._num_packs), 10)

    destinations = dense_values[0]._devices  # pylint: disable=protected-access
    grouped = cross_device_ops._group_value_by_device(dense_values)

    # device_grad_packs:
    # [[(t0_gpu0, None), (t1_gpu0, None)], [(t0_gpu1, None), (t1_gpu1, None)]]
    device_grad_packs, tensor_packer = cross_device_ops._pack_tensors(grouped, self._num_packs)

    reduced = aggregate_gradients(device_grad_packs)

    reduced = cross_device_ops._unpack_tensors(reduced, tensor_packer)
    return cross_device_ops._ungroup_and_make_mirrored(reduced, dense_values[0], reduce_op)
