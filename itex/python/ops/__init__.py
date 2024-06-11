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

import os
# pylint: disable=g-bad-import-order,unused-import,missing-module-docstring,unused-import,line-too-long
from intel_extension_for_tensorflow.python.ops.beam_select import beam_select_kv_cache
from intel_extension_for_tensorflow.python.ops.activations import gelu
from intel_extension_for_tensorflow.python.ops.rotary_embedding import qk_rotary_positional_embedding
from intel_extension_for_tensorflow.python.ops import ops_grad as _ops_grad

from intel_extension_for_tensorflow.python.ops.multi_head_attention import scaled_dot_product_attention

if os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"):
    from intel_extension_for_tensorflow.python.ops.group_norm import GroupNormalization
    from intel_extension_for_tensorflow.python.ops.layer_norm import LayerNormalization
    from intel_extension_for_tensorflow.python.ops.mlp import FusedDenseBiasAddGelu
    from intel_extension_for_tensorflow.python.ops.rms_norm import RMSNormalization
    from intel_extension_for_tensorflow.python.ops.recurrent import ItexLSTM
    from intel_extension_for_tensorflow.python.ops.optimizers import AdamOptimizer, AdamWithWeightDecayOptimizer, AdamWithWeightDecayLegacyOptimizer, LAMBOptimizer
else:
    from intel_extension_for_tensorflow.python.ops.layer_norm_k3 import LayerNormalization
    from intel_extension_for_tensorflow.python.ops.group_norm_k3 import GroupNormalization
    from intel_extension_for_tensorflow.python.ops.mlp_k3 import Dense as FusedDenseBiasAddGelu
    from intel_extension_for_tensorflow.python.ops.optimizers_k3 import Adam
    from intel_extension_for_tensorflow.python.ops.rms_norm_k3 import RMSNormalization
