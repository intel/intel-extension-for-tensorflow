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

#ifndef ITEX_CORE_KERNELS_GPU_FUSED_BINARY_OP_H_
#define ITEX_CORE_KERNELS_GPU_FUSED_BINARY_OP_H_

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

#define MAX_LENGTH 6

enum class UpdateOp { ADD, SUB, SUB1, MUL };
template <typename T, int vec_size>
class FusedBinaryFunctor {
  using Tvec = AlignedVector<T, vec_size>;

 public:
  FusedBinaryFunctor(uint32_t num_elements, int32_t num_inputs, const T* input0,
                     const T* input1, const T* input2, const T* input3,
                     const T* input4, const T* input5, const T* input6,
                     T* output, const UpdateOp ops[MAX_LENGTH],
                     const bool is_scalars[MAX_LENGTH])
      : num_elements_(num_elements),
        num_inputs_(num_inputs),
        input0_(input0),
        input1_(input1),
        input2_(input2),
        input3_(input3),
        input4_(input4),
        input5_(input5),
        input6_(input6),
        output_(output) {
    for (int i = 0; i < num_inputs; ++i) {
      ops_[i] = ops[i];
      is_scalars_[i] = is_scalars[i];
    }
  }
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id * vec_size >= num_elements_) return;
    if ((id + 1) * vec_size > num_elements_) {
      // Has tail value that cannot use vector
      for (auto i = id * vec_size; i < num_elements_; ++i) {
        auto offset0 = is_scalars_[0] ? 0 : i;
        auto offset1 = is_scalars_[1] ? 0 : i;
        T res = cal(ops_[0], input0_[offset0], input1_[offset1]);
        offset0 = is_scalars_[2] ? 0 : i;
        res = cal(ops_[1], res, input2_[offset0]);
        if (num_inputs_ > 3) {
          offset0 = is_scalars_[3] ? 0 : i;
          res = cal(ops_[2], res, input3_[offset0]);
        }
        output_[i] = res;
      }
      return;
    } else {
#if !defined(EIGEN_DONT_VECTORIZE_SYCL) && defined(__SYCL_DEVICE_ONLY__)
      if (std::is_same<T, Eigen::bfloat16>::value) {
        Tvec in0 = get_vec(0, id * vec_size, input0_);
        Tvec in1 = get_vec(1, id * vec_size, input1_);
        Tvec in2 = get_vec(2, id * vec_size, input2_);
        sycl::vec<float, vec_size> in0_fp32 =
            Eigen::bfloat16_impl::Bf16ToF32<vec_size>(
                *reinterpret_cast<sycl::vec<uint16_t, vec_size>*>(&in0));
        sycl::vec<float, vec_size> in1_fp32 =
            Eigen::bfloat16_impl::Bf16ToF32<vec_size>(
                *reinterpret_cast<sycl::vec<uint16_t, vec_size>*>(&in1));
        sycl::vec<float, vec_size> in2_fp32 =
            Eigen::bfloat16_impl::Bf16ToF32<vec_size>(
                *reinterpret_cast<sycl::vec<uint16_t, vec_size>*>(&in2));
        sycl::vec<float, vec_size> res =
            cal(ops_[1], cal(ops_[0], in0_fp32, in1_fp32), in2_fp32);

        if (num_inputs_ > 3) {
          Tvec in3 = get_vec(3, id * vec_size, input3_);
          sycl::vec<float, vec_size> in3_fp32 =
              Eigen::bfloat16_impl::Bf16ToF32<vec_size>(
                  *reinterpret_cast<sycl::vec<uint16_t, vec_size>*>(&in3));
          res = cal(ops_[2], res, in3_fp32);
        }

        sycl::vec<uint16_t, vec_size>* out =
            reinterpret_cast<sycl::vec<uint16_t, vec_size>*>(output_ +
                                                             id * vec_size);
        *out = Eigen::bfloat16_impl::F32ToBf16<vec_size>(res);
        return;
      }
#endif
      Tvec res = cal(ops_[0], get_vec(0, id * vec_size, input0_),
                     get_vec(1, id * vec_size, input1_));
      res = cal(ops_[1], res, get_vec(2, id * vec_size, input2_));

      if (num_inputs_ > 3) {
        res = cal(ops_[2], res, get_vec(3, id * vec_size, input3_));
      }

      Tvec* out = reinterpret_cast<Tvec*>(output_ + id * vec_size);
      *out = res;
    }
  }

 private:
  template <typename U>
  inline U cal(UpdateOp op, U in0, U in1) const {
    switch (op) {
      case UpdateOp::ADD:
        return in0 + in1;
        break;
      case UpdateOp::SUB:
        return in0 - in1;
        break;
      case UpdateOp::SUB1:
        return in1 - in0;
        break;
      case UpdateOp::MUL:
        return in0 * in1;
        break;

      default:
        return U(0);
        break;
    }
  }

  inline Tvec get_vec(int in_idx, int offset, const T* input) const {
    if (is_scalars_[in_idx]) {
      return Tvec(input[0]);
    } else {
      const Tvec* in_vec = reinterpret_cast<const Tvec*>(input + offset);
      return *in_vec;
    }
  }

  const uint32_t num_elements_;
  const int32_t num_inputs_;
  const T* input0_;
  const T* input1_;
  const T* input2_;
  const T* input3_;
  const T* input4_;
  const T* input5_;
  const T* input6_;
  T* output_;
  UpdateOp ops_[MAX_LENGTH];
  bool is_scalars_[MAX_LENGTH];
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FUSED_BINARY_OP_H_
