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
import numpy as np
import tensorflow as tf

common_2d_input_size = [[32, 8192], [33, 8193]]
tailed_no_tailed_size = [8192, 16384 * 16384, 8193, 16385 * 16385]
broadcast_binary_size_x = [[32,16,512,512], [32,16,512,512], [32,16,512,513]]
broadcast_binary_size_y = [[32,1,512,512], [1,1,1,1], [32,16,512,513]]
reduction_size = [[32,16,512,512], [32,1,512,513], [1,16,513,512], [1,1,1,1]]
reduction_axis = [0, 1, 2, 3]
reduction_keepdims = [False, True]

def multi_run(iteration):
  def decorator(func):
    def wrap_func(*args, **kwargs):
      for i in range(iteration):
        func(*args, **kwargs)
    return wrap_func
  return decorator

def add_profiling(func):
  def wrap_func(*args, **kwargs):
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                         python_tracer_level = 1,
                                         device_tracer_level = 1)
    if not os.path.exists("../../profile_data"):
        os.mkdir("../../profile_data")
    tf.profiler.experimental.start('../../profile_data', options = options)
    func(*args, **kwargs)
    tf.profiler.experimental.stop()
  return wrap_func

# we add this to flush L3 in PVC, currently PVC L3 size = 204M
def flush_cache():
  cache_size = 1024 * 1024 * 30
  in_array = np.random.normal(size=[cache_size])
  array = tf.constant(in_array, dtype=tf.float32)
  array = array + 2
