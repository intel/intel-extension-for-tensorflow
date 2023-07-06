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

#include "itex/core/kernels/common/cast_op.h"

#include "itex/core/kernels/gpu/cast_op_impl.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

CAST_FUNCTORS(GPUDevice);

#define DEFINE(O, I) template struct CastFunctor<GPUDevice, O, I>

#ifdef ITEX_ENABLE_DOUBLE
#define DEFINE_ALL_TO(out_type)          \
  DEFINE(out_type, bool);                \
  DEFINE(out_type, uint8);               \
  DEFINE(out_type, uint16);              \
  DEFINE(out_type, uint32);              \
  DEFINE(out_type, uint64);              \
  DEFINE(out_type, int8);                \
  DEFINE(out_type, int16);               \
  DEFINE(out_type, int32);               \
  DEFINE(out_type, int64);               \
  DEFINE(out_type, Eigen::half);         \
  DEFINE(out_type, Eigen::bfloat16);     \
  DEFINE(out_type, float);               \
  DEFINE(out_type, double);              \
  DEFINE(out_type, std::complex<float>); \
  DEFINE(out_type, std::complex<double>);

#else
#define DEFINE_ALL_TO(out_type)      \
  DEFINE(out_type, bool);            \
  DEFINE(out_type, uint8);           \
  DEFINE(out_type, uint16);          \
  DEFINE(out_type, uint32);          \
  DEFINE(out_type, uint64);          \
  DEFINE(out_type, int8);            \
  DEFINE(out_type, int16);           \
  DEFINE(out_type, int32);           \
  DEFINE(out_type, int64);           \
  DEFINE(out_type, Eigen::half);     \
  DEFINE(out_type, Eigen::bfloat16); \
  DEFINE(out_type, float);           \
  DEFINE(out_type, std::complex<float>);
#endif  // ITEX_ENABLE_DOUBLE

DEFINE_ALL_TO(bool);
DEFINE_ALL_TO(uint8);
DEFINE_ALL_TO(uint16);
DEFINE_ALL_TO(uint32);
DEFINE_ALL_TO(uint64);
DEFINE_ALL_TO(int8);
DEFINE_ALL_TO(int16);
DEFINE_ALL_TO(int32);
DEFINE_ALL_TO(int64);
#ifdef ITEX_ENABLE_DOUBLE
DEFINE_ALL_TO(double);
DEFINE_ALL_TO(std::complex<double>);
#endif  // ITEX_ENABLE_DOUBLE

#define DEFINE_ALL_TO_FLOAT(out_type) \
  DEFINE(out_type, bool);             \
  DEFINE(out_type, uint8);            \
  DEFINE(out_type, uint16);           \
  DEFINE(out_type, uint32);           \
  DEFINE(out_type, uint64);           \
  DEFINE(out_type, int8);             \
  DEFINE(out_type, int16);            \
  DEFINE(out_type, int32);            \
  DEFINE(out_type, int64);            \
  DEFINE(out_type, Eigen::half);      \
  DEFINE(out_type, Eigen::bfloat16);  \
  DEFINE(out_type, float);            \
  DEFINE(out_type, std::complex<float>);

#define DEFINE_ALL_TO_HALF(out_type) \
  DEFINE(out_type, bool);            \
  DEFINE(out_type, uint8);           \
  DEFINE(out_type, uint16);          \
  DEFINE(out_type, uint32);          \
  DEFINE(out_type, uint64);          \
  DEFINE(out_type, int8);            \
  DEFINE(out_type, int16);           \
  DEFINE(out_type, int32);           \
  DEFINE(out_type, int64);           \
  DEFINE(out_type, Eigen::half);     \
  DEFINE(out_type, Eigen::bfloat16);

DEFINE_ALL_TO_HALF(Eigen::half);
DEFINE_ALL_TO_HALF(Eigen::bfloat16);
DEFINE_ALL_TO_FLOAT(float);
DEFINE_ALL_TO_FLOAT(std::complex<float>);

#undef DEFINE_ALL_TO_FLOAT
#undef DEFINE_ALL_TO_HALF
#undef DEFINE_ALL_FROM
#undef DEFINE

}  // namespace functor

class GpuCastOp : public OpKernel {
 public:
  explicit GpuCastOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("SrcT", &external_src_dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("DstT", &external_dst_dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("Truncate", &use_truncation_));

    // Quantized data types use the same underlying format as their non
    // quantized version so we use the non quantized implementation for casting.
    if (external_dst_dtype_ == DT_QUINT8) {
      dst_dtype_ = DT_UINT8;
    } else if (external_dst_dtype_ == DT_QINT8) {
      dst_dtype_ = DT_INT8;
    } else if (external_dst_dtype_ == DT_QINT32) {
      dst_dtype_ = DT_INT32;
    } else if (external_dst_dtype_ == DT_QINT16) {
      dst_dtype_ = DT_INT16;
    } else if (external_dst_dtype_ == DT_QUINT16) {
      dst_dtype_ = DT_UINT16;
    } else {
      dst_dtype_ = external_dst_dtype_;
    }

    if (external_src_dtype_ == DT_QUINT8) {
      src_dtype_ = DT_UINT8;
    } else if (external_src_dtype_ == DT_QINT8) {
      src_dtype_ = DT_INT8;
    } else if (external_src_dtype_ == DT_QINT32) {
      src_dtype_ = DT_INT32;
    } else if (external_src_dtype_ == DT_QINT16) {
      src_dtype_ = DT_INT16;
    } else if (external_src_dtype_ == DT_QUINT16) {
      src_dtype_ = DT_UINT16;
    } else {
      src_dtype_ = external_src_dtype_;
    }

    OP_REQUIRES_OK(context, Prepare());
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& inp = context->input(0);
    if (cast_work_ == nullptr) {
      context->set_output(0, inp);
    } else {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, inp.shape(), &out));
      if (inp.NumElements() > 0) {
        cast_work_(*context, inp, out, use_truncation_);
      }
    }
  }

 private:
  Status Unimplemented();

  Status Prepare();

  DataType src_dtype_;
  DataType dst_dtype_;
  DataType external_src_dtype_;
  DataType external_dst_dtype_;
  bool use_truncation_;
  CastFunctorType cast_work_ = nullptr;
};

Status GpuCastOp::Unimplemented() {
  return errors::Unimplemented("Cast ", DataTypeString(external_src_dtype_),
                               " to ", DataTypeString(external_dst_dtype_),
                               " is not supported");
}

Status GpuCastOp::Prepare() {
  // TODO(itex): Support quantize types.
  if (external_src_dtype_ != src_dtype_ || external_dst_dtype_ != dst_dtype_) {
    return Unimplemented();
  } else if (external_src_dtype_ == external_dst_dtype_) {
    cast_work_ = nullptr;  // Identity
    return Status::OK();
  }
  if (src_dtype_ == DT_BOOL) {
    cast_work_ = GetGpuCastFromBool(dst_dtype_);
  } else if (src_dtype_ == DT_UINT8) {
    cast_work_ = GetGpuCastFromUint8(dst_dtype_);
  } else if (src_dtype_ == DT_UINT16) {
    cast_work_ = GetGpuCastFromUint16(dst_dtype_);
  } else if (src_dtype_ == DT_UINT32) {
    cast_work_ = GetGpuCastFromUint32(dst_dtype_);
  } else if (src_dtype_ == DT_UINT64) {
    cast_work_ = GetGpuCastFromUint64(dst_dtype_);
  } else if (src_dtype_ == DT_INT8) {
    cast_work_ = GetGpuCastFromInt8(dst_dtype_);
  } else if (src_dtype_ == DT_INT16) {
    cast_work_ = GetGpuCastFromInt16(dst_dtype_);
  } else if (src_dtype_ == DT_INT32) {
    cast_work_ = GetGpuCastFromInt32(dst_dtype_);
  } else if (src_dtype_ == DT_INT64) {
    cast_work_ = GetGpuCastFromInt64(dst_dtype_);
  } else if (src_dtype_ == DT_HALF) {
    cast_work_ = GetGpuCastFromHalf(dst_dtype_);
  } else if (src_dtype_ == DT_FLOAT) {
    cast_work_ = GetGpuCastFromFloat(dst_dtype_);
  } else if (src_dtype_ == DT_BFLOAT16) {
    cast_work_ = GetGpuCastFromBfloat(dst_dtype_);
  } else if (src_dtype_ == DT_COMPLEX64) {
    cast_work_ = GetGpuCastFromComplex64(dst_dtype_);
  }
#ifdef ITEX_ENABLE_DOUBLE
  if (src_dtype_ == DT_DOUBLE) {
    cast_work_ = GetGpuCastFromDouble(dst_dtype_);
  } else if (src_dtype_ == DT_COMPLEX128) {
    cast_work_ = GetGpuCastFromComplex128(dst_dtype_);
  }
#endif  // ITEX_ENABLE_DOUBLE
  return cast_work_ == nullptr ? Unimplemented() : Status::OK();
}

#ifdef ITEX_ENABLE_DOUBLE
#define CURRY_TYPES2(FN, arg0)    \
  FN(arg0, bool);                 \
  FN(arg0, uint8);                \
  FN(arg0, uint16);               \
  FN(arg0, uint32);               \
  FN(arg0, uint64);               \
  FN(arg0, int8);                 \
  FN(arg0, int16);                \
  FN(arg0, int32);                \
  FN(arg0, int64_t);              \
  FN(arg0, Eigen::half);          \
  FN(arg0, Eigen::bfloat16);      \
  FN(arg0, double);               \
  FN(arg0, float);                \
  FN(arg0, std::complex<double>); \
  FN(arg0, std::complex<float>);
#else
#define CURRY_TYPES2(FN, arg0) \
  FN(arg0, bool);              \
  FN(arg0, uint8);             \
  FN(arg0, uint16);            \
  FN(arg0, uint32);            \
  FN(arg0, uint64);            \
  FN(arg0, int8);              \
  FN(arg0, int16);             \
  FN(arg0, int32);             \
  FN(arg0, int64_t);           \
  FN(arg0, Eigen::half);       \
  FN(arg0, Eigen::bfloat16);   \
  FN(arg0, float);             \
  FN(arg0, std::complex<float>);
#endif

#define REGISTER_CAST_GPU(srctype, dsttype)                    \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_GPU),             \
                          GpuCastOp)

CURRY_TYPES2(REGISTER_CAST_GPU, bool);
CURRY_TYPES2(REGISTER_CAST_GPU, int8);
CURRY_TYPES2(REGISTER_CAST_GPU, int16);
CURRY_TYPES2(REGISTER_CAST_GPU, int32);
CURRY_TYPES2(REGISTER_CAST_GPU, int64);
CURRY_TYPES2(REGISTER_CAST_GPU, uint8);
CURRY_TYPES2(REGISTER_CAST_GPU, uint16);
CURRY_TYPES2(REGISTER_CAST_GPU, uint32);
CURRY_TYPES2(REGISTER_CAST_GPU, uint64);
CURRY_TYPES2(REGISTER_CAST_GPU, Eigen::half);
CURRY_TYPES2(REGISTER_CAST_GPU, Eigen::bfloat16);
CURRY_TYPES2(REGISTER_CAST_GPU, float);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<float>);
#ifdef ITEX_ENABLE_DOUBLE
CURRY_TYPES2(REGISTER_CAST_GPU, double);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<double>);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_CAST_GPU
#undef CURRY_TYPES2
}  // namespace itex
