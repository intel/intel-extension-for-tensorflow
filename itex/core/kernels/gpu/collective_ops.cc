/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/kernels/gpu/collective_ops.h"

#include "itex/core/utils/op_requires.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

// Base class for all collective ops.
// About memory management and stream syncing:
// 1. The manager has a stream for each rank.
// 2. For input tensors to the communicator, the compute stream is passed to the
//    Manager which will do a needed
//    communicator_stream.barrier(input_tensor_stream).
// 3. The done_callback of the async kernel is not called by the
//    Manager until after the collective kernel is submitted. This
//    is enough to a) keep the input tensor data valid for the lifetime of the
//    collective; and b) ensure the data in the output tensor is available
//    when the async op kernel's done callback is called.
class CollectiveOpBase : public AsyncOpKernel {
 public:
  explicit CollectiveOpBase(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("num_devices", &num_devices_));
    OP_REQUIRES_OK(c, c->GetAttr("shared_name", &collective_prefix_));
  }

  string GetCollectiveKey(OpKernelContext* c) {
    return strings::StrCat(collective_prefix_, ";", c->GetStepId(), ";",
                           c->GetFrameId(), ":", c->GetIterId());
  }

  int num_devices() const { return num_devices_; }

 private:
  int num_devices_;
  string collective_prefix_;

  CollectiveOpBase(const CollectiveOpBase&) = delete;
  void operator=(const CollectiveOpBase&) = delete;
};

class ReduceOpBase : public CollectiveOpBase {
 public:
  explicit ReduceOpBase(OpKernelConstruction* c) : CollectiveOpBase(c) {
    string reduction;
    OP_REQUIRES_OK(c, c->GetAttr("reduction", &reduction));
    if (reduction == "min") {
      reduction_op_ = ReductionOp::MIN;
    } else if (reduction == "max") {
      reduction_op_ = ReductionOp::MAX;
    } else if (reduction == "sum") {
      reduction_op_ = ReductionOp::SUM;
    } else if (reduction == "prod") {
      reduction_op_ = ReductionOp::PROD;
    } else {
      OP_REQUIRES_OK(c,
                     errors::InvalidArgument("Invalid reduction: ", reduction));
    }
  }

  ReductionOp reduction_op() const { return reduction_op_; }

 private:
  ReductionOp reduction_op_;
};

class AllReduceOp : public ReduceOpBase {
 public:
  explicit AllReduceOp(OpKernelConstruction* context) : ReduceOpBase(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor* input = &context->input(0);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(context,
                         context->forward_input_or_allocate_output(
                             {0}, 0, input->shape(), &output),
                         done);
    auto actual_done = [context, done](Status s) {
      OP_REQUIRES_OK_ASYNC(context, s, done);
      done();
    };
    auto compute_stream = context->GetDeviceStream();
    int gpu_id = TF_GetDeviceId(context->Get());
    auto participant = std::make_unique<CollectiveManager::Participant>(
        compute_stream, gpu_id, input, output, std::move(actual_done));
    CollectiveManager::instance()->AddToAllReduce(
        std::move(participant), {GetCollectiveKey(context), num_devices(), -1},
        reduction_op());
  }
};

REGISTER_ASYNC_KERNEL_BUILDER(Name("ItexAllReduceSend").Device(DEVICE_GPU),
                              AllReduceOp);

}  // end namespace itex
