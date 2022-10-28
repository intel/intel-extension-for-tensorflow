/* Copyright (c) 2022 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ITEX_CORE_KERNELS_COMMON_SLICE_FUNCTOR_H_
#define ITEX_CORE_KERNELS_COMMON_SLICE_FUNCTOR_H_

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

void IntTensorToInt64Vec(const Tensor& tensor,
                         gtl::InlinedVector<int64, 4>* out);

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
void SharedSliceValidation(OpKernelContext* context,
                           const TensorShape& src_tf_shape,
                           TensorShape* dst_tf_shape, bool* is_identity,
                           bool* slice_dim0,
                           gtl::InlinedVector<int64, 4>* begin,
                           gtl::InlinedVector<int64, 4>* size);

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_SLICE_FUNCTOR_H_
