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

#ifndef ITEX_CORE_KERNELS_CPU_FUSED_BINARY_OP_H_
#define ITEX_CORE_KERNELS_CPU_FUSED_BINARY_OP_H_

#include <vector>

#include "itex/core/kernels/common/cwise_ops_common.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

// The max number of FusedBinary ops, but currently only support three.
const int MAX_LENGTH = 6;
using CPUDevice = Eigen::ThreadPoolDevice;

// The supportive binary ops.
enum class UpdateOp { ADD, SUB, SUB1, MUL };

template <typename Tout, typename Tin>
struct select_binary_op {
  Eigen::internal::scalar_sum_op<Tin, Tin> sum_op;
  Eigen::internal::scalar_difference_op<Tin, Tin> sub_op;
  Eigen::internal::scalar_product_op<Tin, Tin> mul_op;

  const UpdateOp update_op_;

  EIGEN_DEVICE_FUNC inline select_binary_op(const select_binary_op& other) =
      default;

  EIGEN_DEVICE_FUNC inline explicit select_binary_op(UpdateOp op)
      : update_op_(op) {}

  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& a, const Tin& b) const {
    switch (update_op_) {
      case UpdateOp::ADD:
        return sum_op(a, b);
        break;
      case UpdateOp::SUB:
        return sub_op(a, b);
        break;
      case UpdateOp::SUB1:
        return sub_op(b, a);
        break;
      case UpdateOp::MUL:
        return mul_op(a, b);
        break;
      default:
        return sum_op(a, b);
        break;
    }
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a,
                                           const Packet& b) const {
    switch (update_op_) {
      case UpdateOp::ADD:
        return sum_op.packetOp(a, b);
        break;
      case UpdateOp::SUB:
        return sub_op.packetOp(a, b);
        break;
      case UpdateOp::SUB1:
        return sub_op.packetOp(b, a);
        break;
      case UpdateOp::MUL:
        return mul_op.packetOp(a, b);
        break;
      default:
        return sum_op.packetOp(a, b);
        break;
    }
  }
};

// One step binary computation.
template <typename T>
class OneBinaryFunctor {
 public:
  using Tvec = typename TTypes<T>::Flat;

  template <typename U>
  void operator()(const CPUDevice& d, UpdateOp op, const U& in0, const U& in1,
                  U* out) {
    switch (op) {
      case UpdateOp::ADD:
        BinaryCompute<functor::add<T>>(d, in0, in1, out);
        break;
      case UpdateOp::SUB:
        BinaryCompute<functor::sub<T>>(d, in0, in1, out);
        break;
      case UpdateOp::SUB1:
        BinaryCompute<functor::sub<T>>(d, in1, in0, out);
        break;
      case UpdateOp::MUL:
        BinaryCompute<functor::mul<T>>(d, in0, in1, out);
        break;

      default:
        BinaryCompute<functor::add<T>>(d, in0, in1, out);
        break;
    }
  }

 private:
  template <typename Functor>
  inline void BinaryCompute(const CPUDevice& d, const Tvec& in0,
                            const Tvec& in1, Tvec* output) const {
    typedef typename Functor::in_type Tin;    // Input scalar data type.
    typedef typename Functor::out_type Tout;  // Output scalar data type.
    typedef typename Functor::func Binary;

    output->device(d) = in0.binaryExpr(in1, typename Functor::func());
  }
};

// This functor evaluates all expressions only when assigned to the output. So
// the Binary Compute function in it should return expression type, not the
// assigned and calculated tensor type.
template <typename T>
class FusedBinaryLazyComputeFunctor {
 public:
  using Tvec = typename TTypes<T>::Flat;
  using DSizes = Eigen::array<Eigen::DenseIndex, 1>;

  void operator()(const CPUDevice& eigen_device, int num_elements,
                  int32_t num_inputs, const std::vector<Tvec>& inputs, Tvec out,
                  const UpdateOp ops[MAX_LENGTH]) const {
    auto mid_operation = BinaryCompute(
        BinaryCompute(inputs[0], inputs[1], ops[0]), inputs[2], ops[1]);
    if (num_inputs > 3) {
      out.device(eigen_device) =
          BinaryCompute(mid_operation, inputs[3], ops[2]);
    } else {
      out.device(eigen_device) = mid_operation;
    }
  }

 private:
  template <typename LeftExpr, typename RightExpr>
  inline auto BinaryCompute(const LeftExpr& in0, const RightExpr& in1,
                            UpdateOp op) const {
    return in0.binaryExpr(in1, select_binary_op<T, T>(op));
  }
};

// Calculate expressions one by one, and assign the result of each expression to
// an intermediate variable.
template <typename T>
class FusedBinaryEagerComputeFunctor {
 public:
  using Tvec = typename TTypes<T>::Flat;

  void operator()(const CPUDevice& eigen_device, int num_elements,
                  int32_t num_inputs, const std::vector<Tvec>& inputs,
                  Tvec output, const UpdateOp ops[MAX_LENGTH]) const {
    OneBinaryFunctor<T>()(eigen_device, ops[0], inputs[0], inputs[1], &output);
    OneBinaryFunctor<T>()(eigen_device, ops[1], output, inputs[2], &output);

    if (num_inputs > 3) {
      OneBinaryFunctor<T>()(eigen_device, ops[2], output, inputs[3], &output);
    }
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_CPU_FUSED_BINARY_OP_H_
