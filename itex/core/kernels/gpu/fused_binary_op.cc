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

#include "itex/core/kernels/gpu/fused_binary_op.h"

#include <string>
#include <vector>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class FusedBinaryOp : public OpKernel {
 public:
  explicit FusedBinaryOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("input_order", &input_order_));
    // last order always is 0.
    input_order_.push_back(0);
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES(context, input_order_.size() == fused_ops_.size(),
                errors::InvalidArgument(
                    "input_order_ and fused_ops_ must have same size. ",
                    input_order_.size(), " vs ", fused_ops_.size()));

    inputs_.resize(MAX_LENGTH + 1, nullptr);
  }

  void Compute(OpKernelContext* context) override {
    int num = context->num_inputs();
    TensorShape output_shape = context->input(0).shape();

    bool has_zero_input = false;

    for (int i = 0; i < num; ++i) {
      // Fused Binary only support scalar and same input, so output size should
      // same with the largest input.
      if (context->input(i).NumElements() > output_shape.num_elements()) {
        output_shape = context->input(i).shape();
        // for [1, 1] and scalar(with shape[]), output shape should be [1, 1]
      } else if (context->input(i).NumElements() ==
                     output_shape.num_elements() &&
                 context->input(i).dims() > output_shape.dims()) {
        output_shape = context->input(i).shape();
      }

      if (context->input(i).NumElements() == 0) {
        has_zero_input = true;
        output_shape = context->input(i).shape();
        break;
      }

      inputs_[i] = const_cast<T*>(context->input(num - 1 - i).flat<T>().data());
      is_scalars_[i] = context->input(num - 1 - i).NumElements() == 1;
      if (i < input_order_.size()) {
        if (fused_ops_[i] == "Add" || fused_ops_[i] == "AddV2") {
          ops_[num - 2 - i] = UpdateOp::ADD;
        } else if (fused_ops_[i] == "Sub") {
          if (input_order_[i] == 0) {
            ops_[num - 2 - i] = UpdateOp::SUB;
          } else {
            ops_[num - 2 - i] = UpdateOp::SUB1;
          }
        } else if (fused_ops_[i] == "Mul") {
          ops_[num - 2 - i] = UpdateOp::MUL;
        }
      }
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (has_zero_input) return;

    const GPUDevice& d = context->eigen_gpu_device();
    auto& stream = d.stream();
    auto workgroup_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    if (std::is_same<T, float>::value) {
      const int vec_length = 4;
      int out_size_vec = (output_shape.num_elements() / vec_length) + 1;
      auto num_workgroups =
          (out_size_vec + workgroup_size - 1) / workgroup_size;
      stream->submit([&](sycl::handler& cgh) {
        FusedBinaryFunctor<T, 4> task(
            output_shape.num_elements(), num, inputs_[0], inputs_[1],
            inputs_[2], inputs_[3], inputs_[4], inputs_[5], inputs_[6],
            output->flat<T>().data(), ops_, is_scalars_);
        cgh.parallel_for<FusedBinaryFunctor<T, 4>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    } else {
      const int vec_length = 8;
      int out_size_vec = (output_shape.num_elements() / vec_length) + 1;
      auto num_workgroups =
          (out_size_vec + workgroup_size - 1) / workgroup_size;
      bool is_16bits = std::is_same<T, Eigen::bfloat16>::value ||
                       std::is_same<T, Eigen::half>::value;
      OP_REQUIRES(context, is_16bits,
                  errors::InvalidArgument("only support float/bfloat16/half"));
      stream->submit([&](sycl::handler& cgh) {
        FusedBinaryFunctor<T, 8> task(
            output_shape.num_elements(), num, inputs_[0], inputs_[1],
            inputs_[2], inputs_[3], inputs_[4], inputs_[5], inputs_[6],
            output->flat<T>().data(), ops_, is_scalars_);
        cgh.parallel_for<FusedBinaryFunctor<T, 8>>(
            sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                              sycl::range<1>(workgroup_size)),
            task);
      });
    }
  }

 private:
  std::vector<int> input_order_;
  std::vector<string> fused_ops_;
  std::vector<T*> inputs_;
  UpdateOp ops_[MAX_LENGTH];
  bool is_scalars_[MAX_LENGTH];
};

#define REGISTER_FUSEDBINARY_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXFusedBinary").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      FusedBinaryOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_FUSEDBINARY_KERNELS);
#undef REGISTER_TOPK_KERNELS

}  // namespace itex
