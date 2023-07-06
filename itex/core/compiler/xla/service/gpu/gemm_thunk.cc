/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/service/gpu/gemm_thunk.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "itex/core/compiler/xla/primitive_util.h"
#include "itex/core/compiler/xla/service/gpu/matmul_utils.h"
#include "itex/core/compiler/xla/service/gpu/mkl.h"
#include "itex/core/compiler/xla/status_macros.h"
#include "itex/core/compiler/xla/util.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/statusor.h"
#include "itex/core/utils/types.h"

namespace itex_xla {
namespace gpu {
namespace {
struct OneDnnMatMulParams {
  dnnl::memory::dims a_dims;
  dnnl::memory::dims b_dims;
  dnnl::memory::dims c_dims;
  dnnl::memory::dims a_strides;
  dnnl::memory::dims b_strides;
  dnnl::memory::dims c_strides;

  OneDnnMatMulParams(dnnl::memory::dims a_dims, dnnl::memory::dims b_dims,
                     dnnl::memory::dims c_dims, dnnl::memory::dims a_strides,
                     dnnl::memory::dims b_strides, dnnl::memory::dims c_strides)
      : a_dims(std::move(a_dims)),
        b_dims(std::move(b_dims)),
        c_dims(std::move(c_dims)),
        a_strides(std::move(a_strides)),
        b_strides(std::move(b_strides)),
        c_strides(std::move(c_strides)) {}
};

std::unique_ptr<OneDnnMatMulParams> CreateMatMulParams(
    int64_t batch_size, const se::blas::MatrixDescriptor& lhs,
    const se::blas::MatrixDescriptor& rhs,
    const se::blas::MatrixDescriptor& out) {
  dnnl::memory::dims lhs_dims{batch_size, lhs.num_rows, lhs.num_cols};
  dnnl::memory::dims rhs_dims{batch_size, rhs.num_rows, rhs.num_cols};
  dnnl::memory::dims out_dims{batch_size, out.num_rows, out.num_cols};

  auto lhs_strides = itex::CalculateTFStrides(lhs_dims);
  auto rhs_strides = itex::CalculateTFStrides(rhs_dims);
  auto out_strides = itex::CalculateTFStrides(out_dims);
  int idx_last = 2;
  int idx_2nd_last = 1;

  // dst(m,n) = \sigma{src(m,k) * weights(k, n)}
  // lhs_strides holds the strides for each dim, say {24, 12, 4, 1} for
  // src_tensor {1, 2, 3, 4} if adj_x_ is false.
  // If adj_x_ is true, swap the innermost two dims of lhs_strides
  // to {24, 12, 1, 4}, just like set memory::format_tag::abdc
  if (lhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs_dims[idx_last], lhs_dims[idx_2nd_last]);
    std::swap(lhs_strides[idx_last], lhs_strides[idx_2nd_last]);
  }
  if (rhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(rhs_dims[idx_last], rhs_dims[idx_2nd_last]);
    std::swap(rhs_strides[idx_last], rhs_strides[idx_2nd_last]);
  }

  return absl::make_unique<OneDnnMatMulParams>(
      lhs_dims, rhs_dims, out_dims, lhs_strides, rhs_strides, out_strides);
}

template <typename T>
Status DoGemm(int64_t batch_size, const se::blas::MatrixDescriptor& lhs,
              const se::blas::MatrixDescriptor& rhs,
              const se::blas::MatrixDescriptor& out, float alpha, float beta,
              se::Stream* stream, se::ScratchAllocator* scratch_allocator) {
  ITEX_CHECK(out.transpose == se::blas::Transpose::kNoTranspose);
  void* lhs_data = const_cast<void*>(lhs.data.opaque());
  void* rhs_data = const_cast<void*>(rhs.data.opaque());
  void* out_data = const_cast<void*>(out.data.opaque());

  auto params = CreateMatMulParams(batch_size, lhs, rhs, out);

  auto src_md = dnnl::memory::desc(params->a_dims, itex::OneDnnType<T>(),
                                   params->a_strides);
  auto weights_md = dnnl::memory::desc(params->b_dims, itex::OneDnnType<T>(),
                                       params->b_strides);
  auto dst_md = dnnl::memory::desc(params->c_dims, itex::OneDnnType<T>(),
                                   params->c_strides);

  auto dnnl_engine =
      itex::FindOrCreateEngine(stream_executor::gpu::AsGpuStreamValue(stream));
#ifndef ITEX_ONEDNN_3_0
  auto matmul_desc =
      std::make_shared<dnnl::matmul::desc>(src_md, weights_md, dst_md);
#endif

  dnnl::primitive_attr post_ops_attr;
  post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // Set fp32 mode.
  dnnl::fpmath_mode fp32_math_mode = itex::GetFP32MathMode<itex::GPUDevice>();
  if (std::is_same<T, float>::value) {
    post_ops_attr.set_fpmath_mode(fp32_math_mode);
  }

  dnnl::post_ops post_ops = dnnl::post_ops();
  // C = alpha * MatMul(A, B) + beta * C
  if (fabs(alpha - 1.0f) > 1e-6)
#ifdef ITEX_ONEDNN_3_0
    post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
#else
    post_ops.append_eltwise(1, dnnl::algorithm::eltwise_linear, alpha, 0.0f);
#endif
  if (fabs(beta - 0.0f) > 1e-6) post_ops.append_sum(beta);
  post_ops_attr.set_post_ops(post_ops);

#ifdef ITEX_ONEDNN_3_0
  auto matmul_pd = std::make_shared<dnnl::matmul::primitive_desc>(
      dnnl_engine, src_md, weights_md, dst_md, post_ops_attr);
#else
  auto matmul_pd = std::make_shared<dnnl::matmul::primitive_desc>(
      *matmul_desc, post_ops_attr, dnnl_engine);
#endif
  std::unordered_map<int, dnnl::memory> fwd_primitive_args;

  void* workspace;
  size_t scratchpad_size = matmul_pd->scratchpad_desc().get_size();
  TF_RETURN_IF_ERROR(
      se::AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));

  auto scratchpad_mem =
      dnnl::memory(matmul_pd->scratchpad_desc(), dnnl_engine, workspace);

  auto matmul_primitive = dnnl::matmul(*matmul_pd);

  auto dnnl_stream = dnnl::sycl_interop::make_stream(
      dnnl_engine, *(stream_executor::gpu::AsGpuStreamValue(stream)));
  auto src_mem = itex::CreateDnnlMemory(src_md, dnnl_engine, lhs_data);

  auto wei_mem = itex::CreateDnnlMemory(weights_md, dnnl_engine, rhs_data);
  auto dst_mem = itex::CreateDnnlMemory(dst_md, dnnl_engine, out_data);
  fwd_primitive_args.emplace(DNNL_ARG_SRC, src_mem);
  fwd_primitive_args.emplace(DNNL_ARG_WEIGHTS, wei_mem);
  fwd_primitive_args.emplace(DNNL_ARG_DST, dst_mem);
  fwd_primitive_args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem);
  matmul_primitive.execute(dnnl_stream, fwd_primitive_args);
  return Status::OK();
}
}  // namespace

GemmThunk::GemmThunk(ThunkInfo thunk_info, GemmConfig config,
                     const BufferAllocation::Slice& lhs_buffer,
                     const BufferAllocation::Slice& rhs_buffer,
                     const BufferAllocation::Slice& output_buffer)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer) {}

Status GemmThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto get_device_address = [&](const BufferAllocation::Slice& slice) {
    return params.buffer_allocations->GetDeviceAddress(slice);
  };

  se::DeviceMemoryBase lhs_data = get_device_address(lhs_buffer_);
  se::DeviceMemoryBase rhs_data = get_device_address(rhs_buffer_);
  se::DeviceMemoryBase output_data = get_device_address(output_buffer_);

  auto& buffer_allocations = *params.buffer_allocations;
  se::ScratchAllocator scratch_allocator(buffer_allocations.device_ordinal(),
                                         buffer_allocations.memory_allocator());

  ITEX_VLOG(3) << "Running GEMM thunk";
  return RunGemm(config_, lhs_data, rhs_data, output_data, params.stream,
                 &scratch_allocator);
}

Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream* stream,
               se::ScratchAllocator* scratch_allocator) {
  ITEX_VLOG(2) << "Executing a GemmThunk";
  se::blas::MatrixDescriptor lhs = GetMatrixDesc(config.lhs_layout, lhs_buffer);
  se::blas::MatrixDescriptor rhs = GetMatrixDesc(config.rhs_layout, rhs_buffer);
  se::blas::MatrixDescriptor output =
      GetMatrixDesc(config.output_layout, output_buffer);
  int64_t batch_size = config.output_layout.batch_size;

  // TODO(cjfj): Support transposed output when using cuBLASLt.
  MakeBlasGemmCompatible(lhs, rhs, output);

  switch (config.output_layout.dtype) {
    case F16:
      return DoGemm<Eigen::half>(batch_size, lhs, rhs, output,
                                 config.alpha.real(), config.beta, stream,
                                 scratch_allocator);
    case BF16:
      return DoGemm<Eigen::bfloat16>(batch_size, lhs, rhs, output,
                                     config.alpha.real(), config.beta, stream,
                                     scratch_allocator);
    case F32:
      return DoGemm<float>(batch_size, lhs, rhs, output, config.alpha.real(),
                           config.beta, stream, scratch_allocator);
    case S32:
    case F64:
    case C64:
    case C128:
    default:
      return InternalError("Unexpected GEMM dtype: %s",
                           primitive_util::LowercasePrimitiveTypeName(
                               config.output_layout.dtype));
  }
}

}  // namespace gpu
}  // namespace itex_xla
