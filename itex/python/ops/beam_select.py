# Copyright (c) 2023 Intel Corporation
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
import os
if os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"):
    from tf_keras.utils import register_keras_serializable
else:
    from keras.src.saving.object_registration import register_keras_serializable
import tensorflow as tf

from tensorflow.python.framework import ops
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python.framework import config


@register_keras_serializable(package="Itex")
def beam_select_kv_cache(cache, indices, input_length=0, name=None):
    if config.list_logical_devices('XPU'):
        with ops.name_scope(name, "beam_select_kv_cache", [cache, indices, input_length]):
            cache = ops.convert_to_tensor(cache, name="cache")
            indices = ops.convert_to_tensor(indices, name="indices")
            return load_ops_library.beam_select_kv_cache(cache, indices, input_length=input_length)
    else:
        return tf.gather(params=cache, indices=indices, axis=1, batch_dims=1)
