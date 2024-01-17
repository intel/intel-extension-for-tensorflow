/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/common/einsum_op_impl.h"

#include "itex/core/kernels/gpu/linalg/fused_einsum_helper.h"

namespace itex {

namespace functor {

template <typename T>
void FusedEinsum<T>::operator()(
    typename FusedEinsum<T>::Arguments& args) {  // NOLINT
  einsum_dispatcher::Dispatch<T>(args);
}

}  // namespace functor

using TransformMeta = std::pair<string, std::vector<std::vector<int>>>;
absl::flat_hash_map<string, TransformMeta>* GetEinsumEquationTransformation() {
  // Record transformation pattern, the definition is illustrated as below:
  // The key is the einsum normalized equation.
  // The value means the dimension mapping rule of lhs, rhs and output tensor
  // after transformation. For example, we can see "abc,abde->dec" convert to
  // "ab,ac->cb" in
  // {"abc,abde->dec", {"ab,ac->cb", {{0, 1}, {2}, {0, 1}, {2}, {3, 1}}}},
  // followed by an array describing the dimension transformation. The first
  // element of this array is {0, 1}, means lhs dim 0 and lhs dim1 merge to a
  // new lhs dim. New lhs dim 2 is same as before. rhs transformation
  // corresponds to the next two arrays, map rule has no change. And the last
  // element {3, 1} describe how to get output dims. 3 means output dim0 is the
  // new rhs dim1, due to the third element of the array belongs to rhs, 1 and
  // so on.
  static absl::flat_hash_map<string, TransformMeta>* transformation =
      new absl::flat_hash_map<string, TransformMeta>{
          // {"ab,ac->cb", {"ab,ac->cb", {{0}, {1}, {0}, {1}, {3, 1}}}},
          // {"abc,abde->dec", {"ab,ac->cb", {{0, 1}, {2}, {0, 1}, {2}, {3,
          // 1}}}},
          // {"abcd,abe->ecd", {"ab,ac->cb", {{0, 1}, {2, 3}, {0, 1}, {2}, {3,
          // 1}}}},
          {"abcd,aecd->aceb",
           {"abcd,aecd->aceb",
            {{0}, {1}, {2}, {3}, {0}, {1}, {2}, {3}, {0, 2, 5, 1}}}},
          {"abcd,aecd->acbe",
           {"abcd,aecd->acbe",
            {{0}, {1}, {2}, {3}, {0}, {1}, {2}, {3}, {0, 2, 1, 5}}}},
          {"abcd,adbe->acbe",
           {"abcd,adbe->acbe",
            {{0}, {1}, {2}, {3}, {0}, {1}, {2}, {3}, {0, 2, 1, 7}}}},
          // {"abcd,acbe->adbe", {"abcd,acbe->adbe", {{0}, {1}, {2}, {3}, {0},
          // {1}, {2}, {3}, {0, 3, 1, 7}}}},
          // {"abcd,acbe->aecd", {"abcd,acbe->aecd", {{0}, {1}, {2}, {3}, {0},
          // {1}, {2}, {3}, {0, 7, 2, 3}}}},
      };
  return transformation;
}

absl::flat_hash_map<string, std::vector<std::vector<int>>>*
GetEinsumOptimizedPattern() {
  // Fused einsum support lhs w/o tranpose + rhs w/o transpose + contraction
  // operend w/o out transpose. For contraction operend, there exists some
  // limitations:
  //  (1) non-broadcast batch matmul
  //  (2) only has 1 free dim and 1 contract dim.
  //  (3) dims should <= 4 and >= 2.
  //  (4) lhs and rhs have same dims.
  //  (5) one of the last two dims should has stride as 1.
  // If some patterns which is supported by fused einsum but not exist in the
  // below map, please add it. This map store equation-permutations key value
  // pair.
  static absl::flat_hash_map<string, std::vector<std::vector<int>>>* result =
      new absl::flat_hash_map<string, std::vector<std::vector<int>>>{
          {"abcd,aecd->aceb", {{0, 2, 1, 3}, {0, 2, 3, 1}, {0, 1, 3, 2}}},
          {"abcd,aecd->acbe", {{0, 2, 1, 3}, {0, 2, 3, 1}, {0, 1, 2, 3}}},
          {"abcd,adbe->acbe", {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 2, 1, 3}}},
          // {"abcd,acbe->aecd", {{0, 2, 3, 1}, {0, 1, 2, 3}, {0, 2, 3, 1}}},
          // {"abcd,acbe->adbe", {{0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 1, 3}}},
          // {"ab,ac->cb", {{1, 0}, {1, 0}, {1, 0}}},
      };
  return result;
}

typedef enum {
  INPUT0 = 0,
  INPUT1 = 1,
  OUTPUT = 2,
} TensorPos;

string NormalizeEquation(string& equation,             // NOLINT
                         OperandLabels& input_labels,  // NOLINT
                         Labels& out_label) {          // NOLINT
  gtl::InlinedVector<string, 2> input_str;
  string output_str;
  Status status = ParseEinsumEquation(equation, &input_str, &output_str);
  if (!status.ok()) return equation;
  int num_inputs = input_str.size();
  int id = 0;
  for (int i = 0; i < num_inputs; ++i) {
    std::for_each(input_str[i].begin(), input_str[i].end(),
                  [&](char& c) { c = 'a' + input_labels[i][id++]; });
    id = 0;
  }
  std::for_each(output_str.begin(), output_str.end(),
                [&](char& c) { c = 'a' + out_label[id++]; });
  string normalized_equation =
      input_str[0] + "," + input_str[1] + "->" + output_str;
  return normalized_equation;
}

template <typename Device, typename T,
          std::enable_if_t<std::is_same_v<Device, GPUDevice>, bool> = true>
bool MayFuseEinsum(OpKernelContext* ctx, string& equation,            // NOLINT
                   OperandLabels& input_labels, Labels& out_label) {  // NOLINT
  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  auto* stream = d.stream();
  auto device = stream->get_device();
  // Fused kernel support on HPC platform with XMX only.
  if (!IsXeHPC(&device) || !HasXMX(&device)) return false;

  // The num of inputs should be 2.
  if (ctx->num_inputs() != 2) return false;
  const Tensor& lhs = ctx->input(0);
  const Tensor& rhs = ctx->input(1);

  // Normalized equation to support more pattern.
  string normalized_equation =
      NormalizeEquation(equation, input_labels, out_label);

  auto transformation = GetEinsumEquationTransformation();

  // Get transformation mete to help reshape tensor.
  if (!transformation->contains(normalized_equation)) return false;
  TransformMeta& transform_meta = transformation->at(normalized_equation);

  auto optimized_pattern = GetEinsumOptimizedPattern();
  // We don't optimize einsum for this equation.
  if (!optimized_pattern->contains(transform_meta.first)) return false;

  std::vector<std::vector<int>>& permutations =
      optimized_pattern->at(transform_meta.first);

  // Permute lhs, rhs and out shape.
  auto permute_func = [](std::vector<int>& shape, std::vector<int>& permute,
                         std::vector<int>& after_permute) -> void {
    for (int i = 0; i < permute.size(); ++i) {
      after_permute.emplace_back(shape[permute[i]]);
    }
  };

  // Calculate strides.
  auto stride_func = [](std::vector<int>& shape, std::vector<int>& permute,
                        std::vector<int>& strides) -> void {
    int stride = 1;
    std::vector<int> temp(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
      temp[i] = stride;
      stride *= shape[i];
    }
    for (int i = 0; i < shape.size(); ++i) {
      strides.emplace_back(temp[permute[i]]);
    }
  };

  // Check if meet xetla limitation.
  static constexpr int STRIDE_ALIGNMENT = 8 / sizeof(T);
  static constexpr int MEMORY_ALIGNMENT = 64 / sizeof(T);
  static constexpr int MAX_OFFSET = (1 << 25) / sizeof(T);
  static constexpr int MIN_ELEMENTS = 64 / sizeof(T);
  auto check_valid = [](std::vector<int>& strides, std::vector<int>& shape,
                        std::pair<int, int>& check_later) -> bool {
    bool stride_align = true;
    int stride_size = strides.size();
    int remained_offset = 0;
    int last_aligned_batch = -1;
    for (int i = 0; i < stride_size; ++i) {
      stride_align = (stride_align) &&
                     (strides[i] == 1 || strides[i] % STRIDE_ALIGNMENT == 0);
      // Batch dims.
      if (stride_size - i > 2) {
        if (strides[i] % MEMORY_ALIGNMENT == 0) {
          // Mark for aligned batch dim.
          last_aligned_batch = i;
        } else {
          // Calculate offset except aligned batch dim.
          remained_offset += strides[i] * (shape[i] - 1);
        }
      }
    }
    check_later.first = last_aligned_batch;
    check_later.second = remained_offset;
    int max_stride =
        std::max(strides[stride_size - 1], strides[stride_size - 2]);
    int min_stride =
        std::min(strides[stride_size - 1], strides[stride_size - 2]);
    remained_offset += max_stride;
    return stride_align && remained_offset < MAX_OFFSET &&
           max_stride >= MIN_ELEMENTS && min_stride == 1;
  };

  // Assign shape according to transform meta.
  auto assign_shape = [&](std::vector<int>& map_ids,
                          const TensorShape& before_transform,
                          std::vector<int>& after_transform) {
    after_transform.emplace_back(before_transform.dim_size(map_ids[0]));
    std::for_each(map_ids.begin() + 1, map_ids.end(), [&](int val) {
      after_transform.back() *= before_transform.dim_size(val);
    });
  };

  std::vector<int> lhs_shape, rhs_shape;
  TensorShape result_shape;
  int dims_after_transform = (transform_meta.second.size() - 1) / 2;
  // Assign inputs shape.
  for (int i = 0; i < dims_after_transform; ++i) {
    assign_shape(transform_meta.second[i], lhs.shape(), lhs_shape);
    assign_shape(transform_meta.second[i + dims_after_transform], rhs.shape(),
                 rhs_shape);
  }
  // Assign output shape.
  auto& out_transform_meta = transform_meta.second.back();
  for (int i = 0; i < out_transform_meta.size(); ++i) {
    int ind = out_transform_meta[i];
    std::for_each(transform_meta.second[ind].begin(),
                  transform_meta.second[ind].end(), [&](int val) {
                    if (ind < dims_after_transform) {
                      result_shape.AddDim(lhs.dim_size(val));
                    } else {
                      result_shape.AddDim(rhs.dim_size(val));
                    }
                  });
  }

  std::vector<int> lhs_permute, rhs_permute;
  std::vector<int> lhs_stride, rhs_stride;
  std::pair<int, int> lhs_check_later, rhs_check_later;
  // lhs and rhs logical shape.
  permute_func(lhs_shape, permutations[TensorPos::INPUT0], lhs_permute);
  permute_func(rhs_shape, permutations[TensorPos::INPUT1], rhs_permute);
  // lhs and rhs physical stride.
  stride_func(lhs_shape, permutations[TensorPos::INPUT0], lhs_stride);
  stride_func(rhs_shape, permutations[TensorPos::INPUT1], rhs_stride);
  // Should meet xetla requirement.
  bool lhs_valid = check_valid(lhs_stride, lhs_permute, lhs_check_later);
  bool rhs_valid = check_valid(rhs_stride, rhs_permute, rhs_check_later);
  if (!lhs_valid || !rhs_valid) return false;

  std::vector<int> out_shape, out_permute, out_stride;
  std::pair<int, int> out_check_later;
  int total_dims = lhs_permute.size();
  int batch_dims = total_dims - 2;
  for (int i = 0; i < batch_dims; ++i) out_shape.emplace_back(lhs_permute[i]);
  out_shape.emplace_back(lhs_permute[total_dims - 2]);
  out_shape.emplace_back(rhs_permute[total_dims - 1]);
  permute_func(out_shape, permutations[TensorPos::OUTPUT], out_permute);
  std::vector<int> permute_inv(out_permute.size());
  for (int i = 0; i < out_permute.size(); ++i) {
    permute_inv[permutations[TensorPos::OUTPUT][i]] = i;
  }
  stride_func(out_permute, permute_inv, out_stride);
  if (!check_valid(out_stride, out_shape, out_check_later)) return false;

  int min_batch =
      std::min(out_check_later.first,
               std::min(lhs_check_later.first, rhs_check_later.first));
  auto update_remained_offset = [&](std::pair<int, int>& check,
                                    std::vector<int>& stride,
                                    std::vector<int>& shape) -> bool {
    for (int i = min_batch + 1; i <= check.first; ++i) {
      check.second += stride[i] * (shape[i] - 1);
    }
    return check.second < MAX_OFFSET;
  };
  if (!update_remained_offset(lhs_check_later, lhs_stride, lhs_shape) ||
      !update_remained_offset(rhs_check_later, rhs_stride, rhs_shape) ||
      !update_remained_offset(out_check_later, out_stride, out_shape)) {
    return false;
  }

  typename functor::FusedEinsum<T>::Arguments args(
      ctx, lhs.flat<T>().data(), lhs_permute, lhs_stride, rhs.flat<T>().data(),
      rhs_permute, rhs_stride, out_shape, out_stride, min_batch, result_shape);
  functor::FusedEinsum<T>()(args);
  return args.finish_;
}

#define REGISTER_EINSUM(D, TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Einsum").Device(DEVICE_##D).TypeConstraint<TYPE>("T"), \
      EinsumOp<D##Device, TYPE>);

#define REGISTER_GPU(TYPE) REGISTER_EINSUM(GPU, TYPE)
TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_half(REGISTER_GPU);
#undef REGISTER_GPU

#undef REGISTER_EINSUM

}  // namespace itex
