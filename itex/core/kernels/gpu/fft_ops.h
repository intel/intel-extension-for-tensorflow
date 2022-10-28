/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_FFT_OPS_H_
#define ITEX_CORE_KERNELS_GPU_FFT_OPS_H_
#if ITEX_USE_MKL
#include <vector>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {
class FFTBase : public OpKernel {
 public:
  explicit FFTBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    const TensorShape& input_shape = in.shape();
    const int fft_rank = Rank();
    OP_REQUIRES(
        ctx, input_shape.dims() >= fft_rank,
        errors::InvalidArgument("Input must have rank of at least ", fft_rank,
                                " but got: ", input_shape.DebugString()));

    Tensor* out;
    TensorShape output_shape = input_shape;
    uint64 fft_shape[3] = {0, 0, 0};

    // In R2C or C2R mode, we use a second input to specify the FFT length
    // instead of inferring it from the input shape.
    if (IsReal()) {
      const Tensor& fft_length = ctx->input(1);
      OP_REQUIRES(ctx,
                  fft_length.shape().dims() == 1 &&
                      fft_length.shape().dim_size(0) == fft_rank,
                  errors::InvalidArgument("fft_length must have shape [",
                                          fft_rank, "]"));

      auto fft_length_as_vec = fft_length.vec<int32>();
      for (int i = 0; i < fft_rank; ++i) {
        fft_shape[i] = fft_length_as_vec(i);
        // Each input dimension must have length of at least fft_shape[i]. For
        // IRFFTs, the inner-most input dimension must have length of at least
        // fft_shape[i] / 2 + 1.
        bool inner_most = (i == fft_rank - 1);
        uint64 min_input_dim_length =
            !IsForward() && inner_most ? fft_shape[i] / 2 + 1 : fft_shape[i];
        auto input_index = input_shape.dims() - fft_rank + i;
        OP_REQUIRES(
            ctx,
            // We pass through empty tensors, so special case them here.
            input_shape.dim_size(input_index) == 0 ||
                input_shape.dim_size(input_index) >= min_input_dim_length,
            errors::InvalidArgument(
                "Input dimension ", input_index,
                " must have length of at least ", min_input_dim_length,
                " but got: ", input_shape.dim_size(input_index)));
        uint64 dim = IsForward() && inner_most && fft_shape[i] != 0
                         ? fft_shape[i] / 2 + 1
                         : fft_shape[i];
        output_shape.set_dim(output_shape.dims() - fft_rank + i, dim);
      }
    } else {
      for (int i = 0; i < fft_rank; ++i) {
        fft_shape[i] =
            output_shape.dim_size(output_shape.dims() - fft_rank + i);
      }
    }

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (IsReal()) {
      if (IsForward()) {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_FLOAT && out->dtype() == DT_COMPLEX64) ||
                (in.dtype() == DT_DOUBLE && out->dtype() == DT_COMPLEX128),
            errors::InvalidArgument("Wrong types for forward real FFT: in=",
                                    in.dtype(), " out=", out->dtype()));
      } else {
        // Backward
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_FLOAT) ||
                (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_DOUBLE),
            errors::InvalidArgument("Wrong types for backward real FFT: in=",
                                    in.dtype(), " out=", out->dtype()));
      }
    } else {
      // Complext to Complex
      OP_REQUIRES(
          ctx,
          (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_COMPLEX64) ||
              (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_COMPLEX128),
          errors::InvalidArgument("Wrong types for FFT: in=", in.dtype(),
                                  " out=", out->dtype()));
    }

    if (input_shape.num_elements() == 0) {
      ITEX_DCHECK_EQ(0, output_shape.num_elements());
      return;
    }

    DoFFT(ctx, in, fft_shape, out);
  }

 protected:
  virtual int Rank() const = 0;
  virtual bool IsForward() const = 0;
  virtual bool IsReal() const = 0;

  // The function that actually computes the FFT.
  virtual void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
                     Tensor* out) = 0;
};

}  // namespace itex
#endif  // ITEX_USE_MKL
#endif  // ITEX_CORE_KERNELS_GPU_FFT_OPS_H_
