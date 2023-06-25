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

#include "itex/core/kernels/cpu/fused_binary_op.h"

#include <string>
#include <vector>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename T>
class FusedBinaryOpCPU : public OpKernel {
 public:
  using Tvec = typename TTypes<T>::Flat;

  explicit FusedBinaryOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("input_order", &input_order_));
    // last order always is 0.
    input_order_.push_back(0);
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES(context, input_order_.size() == fused_ops_.size(),
                errors::InvalidArgument(
                    "input_order_ and fused_ops_ must have same size. ",
                    input_order_.size(), " vs ", fused_ops_.size()));
  }

  void Compute(OpKernelContext* context) override {
    int num_inputs = context->num_inputs();
    TensorShape output_shape = context->input(0).shape();

    int total_elements = 0;
    std::vector<Tvec> inputs;
    Tensor* output = nullptr;
    UpdateOp ops[MAX_LENGTH];

    for (int i = 0; i < num_inputs; ++i) {
      // Fused Binary only support scalar and same input, so output size should
      // same with the largest input.
      const auto& shape = context->input(num_inputs - 1 - i).shape();
      int elems_i = shape.num_elements();

      // Directly return if nothing to do.
      if (elems_i == 0) {
        output_shape = shape;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, output_shape, &output));
        return;
      }

      if (elems_i > output_shape.num_elements()) {
        output_shape = shape;
        // for [1, 1] and scalar(with shape[]), output shape should be [1, 1]
      } else if (elems_i == output_shape.num_elements() &&
                 shape.dims() > output_shape.dims()) {
        output_shape = shape;
      }

      total_elements += elems_i;

      inputs.push_back(
          const_cast<Tensor&>(context->input(num_inputs - 1 - i)).flat<T>());
      ITEX_CHECK(!TensorShapeUtils::IsScalar(shape))
          << "Unsupported shape in FusedBinary: " << shape.DebugString();

      if (i < input_order_.size()) {
        if (fused_ops_[i] == "Add" || fused_ops_[i] == "AddV2") {
          ops[num_inputs - 2 - i] = UpdateOp::ADD;
        } else if (fused_ops_[i] == "Sub") {
          if (input_order_[i] == 0) {
            ops[num_inputs - 2 - i] = UpdateOp::SUB;
          } else {
            ops[num_inputs - 2 - i] = UpdateOp::SUB1;
          }
        } else if (fused_ops_[i] == "Mul") {
          ops[num_inputs - 2 - i] = UpdateOp::MUL;
        }
      }
      if (i >= 1) {
        total_elements += output_shape.num_elements();
      }
    }

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    int num_elements = output_shape.num_elements();
    const CPUDevice& eigen_device = context->eigen_device<CPUDevice>();

    // Typical L1 cache size per core: 47kb. This is an empirical value. When
    // the data size exceeds this value, use Eigen's lazy computing method,
    // which evaluates a series of expressions only at the end when they are
    // assigned to the output. The opposite is the eager method, which
    // immediately evaluates an expression and assigns the result to an
    // intermediate variable. The lazy mode can reduce memory overhead and
    // greatly improve the speed, but when the data is small the extra overhead
    // it introduces is greater than the benefits, so in this case we turn back
    // to the eager implementation.
    const int kCacheSize = 47321;
    bool enable_lazy_computing =
        (total_elements / eigen_device.numThreads() * sizeof(T) > kCacheSize);

    if (enable_lazy_computing) {
      FusedBinaryLazyComputeFunctor<T>()(eigen_device, num_elements, num_inputs,
                                         inputs, output->flat<T>(), ops);
    } else {
      FusedBinaryEagerComputeFunctor<T>()(eigen_device, num_elements,
                                          num_inputs, inputs, output->flat<T>(),
                                          ops);
    }
  }

 private:
  std::vector<int> input_order_;
  std::vector<string> fused_ops_;
};

#define REGISTER_FUSEDBINARY_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXFusedBinary").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      FusedBinaryOpCPU<type>)

TF_CALL_CPU_NUMBER_TYPES(REGISTER_FUSEDBINARY_KERNELS);
#undef REGISTER_FUSEDBINARY_KERNELS

}  // namespace itex
