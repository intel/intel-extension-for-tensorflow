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

#include <algorithm>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct CheckNumericsLaunch {
  void Run(const GPUDevice& d, const T* data, int size,
           int abnormal_detected[2]);
};

extern template struct CheckNumericsLaunch<Eigen::half>;
extern template struct CheckNumericsLaunch<Eigen::bfloat16>;
extern template struct CheckNumericsLaunch<float>;

template <typename T>
struct CheckNumericsV2Launch {
  void Run(const GPUDevice& d, const T* data, int size,
           int abnormal_detected[3]);
};

extern template struct CheckNumericsV2Launch<Eigen::half>;
extern template struct CheckNumericsV2Launch<Eigen::bfloat16>;
extern template struct CheckNumericsV2Launch<float>;
extern template struct CheckNumericsV2Launch<double>;

namespace {

template <typename Device, typename T>
class CheckNumericsOp;

// Partial specialization for GPU
template <typename T>
class CheckNumericsOp<GPUDevice, T> : public OpKernel {
 public:
  explicit CheckNumericsOp(OpKernelConstruction* context) : OpKernel(context) {
    // message_ is used as the prefix for the assertion error message. For
    // instance, this can be the name of the input op that produced the tensor.
    OP_REQUIRES_OK(context, context->GetAttr("message", &message_));
  }

  void Compute(OpKernelContext* context) override {
    // pass along the input to the output
    context->set_output(0, context->input(0));
    if (context->input(0).NumElements() == 0) {
      return;
    }
    auto input = context->input(0).flat<T>();

    // Allocate and initialize the elements to hold the check results
    const int abnormal_detected_size = 2;
    Tensor abnormal_detected;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({abnormal_detected_size}),
                                &abnormal_detected));

    auto* stream = context->GetDeviceStream();

    OP_REQUIRES(context, stream != nullptr,
                errors::Internal("No GPU stream available."));

    const GPUDevice& d = context->eigen_device<GPUDevice>();

    auto abnormal_detected_ptr = abnormal_detected.flat<int>().data();

    stream->fill<int>(abnormal_detected_ptr, 0,
                      abnormal_detected.flat<int>().size());
    // Call the SYCL kernels for the numerical checks
    CheckNumericsLaunch<T>().Run(d, input.data(), input.size(),
                                 abnormal_detected.flat<int>().data());
    // Copy the results from device to host
    AllocatorAttributes attr;
    attr.set_on_host(true);
    // attr.set_gpu_compatible(true);
    Tensor abnormal_detected_host;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({abnormal_detected_size}),
                                &abnormal_detected_host, attr));

    stream
        ->memcpy(abnormal_detected_host.flat<int>().data(),
                 abnormal_detected_ptr, abnormal_detected_size * sizeof(int))
        .wait();

    // We have observed crashes on some network stacks when not holding
    // this tensor reference.
    // TensorReference abnormal_detected_ref(abnormal_detected);
    auto abnormal_detected_host_flat = abnormal_detected_host.flat<int>();
    int is_nan = abnormal_detected_host_flat(0);
    int is_inf = abnormal_detected_host_flat(1);
    // abnormal_detected_ref.Unref();
    if (is_nan || is_inf) {
      string status;
      ITEX_LOG(ERROR) << "abnormal_detected_host @"
                      << abnormal_detected_host_flat.data() << " = {" << is_nan
                      << ", " << is_inf << "} " << message_;

      // Results should always be 1 or 0.  If we see anything else then
      // there has been some GPU memory corruption.
      ITEX_CHECK_GE(is_nan, 0);
      ITEX_CHECK_GE(is_inf, 0);
      ITEX_CHECK_LE(is_nan, 1);
      ITEX_CHECK_LE(is_inf, 1);

      if (is_nan && is_inf) {
        status = "Inf and NaN";
      } else if (is_nan) {
        status = "NaN";
      } else if (is_inf) {
        status = "Inf";
      }
      context->SetStatus(errors::InvalidArgument(message_, " : Tensor had ",
                                                 status, " values"));
    }
  }

 private:
  string message_;
};

template <typename Device, typename T>
class CheckNumericsV2Op;

// The v2 op differs from the v1 in that it distinguishes -inf and +inf.
template <typename T>
class CheckNumericsV2Op<GPUDevice, T> : public OpKernel {
 public:
  explicit CheckNumericsV2Op(OpKernelConstruction* context)
      : OpKernel(context) {
    // message_ is used as the prefix for the assertion error message. For
    // instance, this can be the name of the input op that produced the tensor.
    OP_REQUIRES_OK(context, context->GetAttr("message", &message_));
  }

  void Compute(OpKernelContext* context) override {
    // pass along the input to the output
    context->set_output(0, context->input(0));
    if (context->input(0).NumElements() == 0) {
      return;
    }
    auto input = context->input(0).flat<T>();

    // Allocate and initialize the elements to hold the check results
    const int abnormal_detected_size = 3;
    Tensor abnormal_detected;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({abnormal_detected_size}),
                                &abnormal_detected));

    auto* stream = context->GetDeviceStream();

    OP_REQUIRES(context, stream != nullptr,
                errors::Internal("No GPU stream available."));

    const GPUDevice& d = context->eigen_device<GPUDevice>();

    auto abnormal_detected_ptr = abnormal_detected.flat<int>().data();

    stream->fill<int>(abnormal_detected_ptr, 0,
                      abnormal_detected.flat<int>().size());
    // Call the SYCL kernels for the numerical checks
    CheckNumericsV2Launch<T>().Run(d, input.data(), input.size(),
                                   abnormal_detected.flat<int>().data());
    // Copy the results from device to host
    AllocatorAttributes attr;
    attr.set_on_host(true);
    // attr.set_gpu_compatible(true);
    Tensor abnormal_detected_host;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({abnormal_detected_size}),
                                &abnormal_detected_host, attr));

    stream
        ->memcpy(abnormal_detected_host.flat<int>().data(),
                 abnormal_detected_ptr, abnormal_detected_size * sizeof(int))
        .wait();

    // We have observed crashes on some network stacks when not holding
    // this tensor reference.
    // TensorReference abnormal_detected_ref(abnormal_detected);
    auto abnormal_detected_host_flat = abnormal_detected_host.flat<int>();
    int is_nan = abnormal_detected_host_flat(0);
    int is_negative_inf = abnormal_detected_host_flat(1);
    int is_positive_inf = abnormal_detected_host_flat(2);

    // Results should always be 1 or 0.  If we see anything else then
    // there has been some GPU memory corruption.
    ITEX_CHECK_GE(is_nan, 0);
    ITEX_CHECK_GE(is_negative_inf, 0);
    ITEX_CHECK_GE(is_positive_inf, 0);
    ITEX_CHECK_LE(is_nan, 1);
    ITEX_CHECK_LE(is_negative_inf, 1);
    ITEX_CHECK_LE(is_positive_inf, 1);

    // abnormal_detected_ref.Unref();
    if (is_nan || is_negative_inf || is_positive_inf) {
      std::vector<string> anomalies;
      if (is_negative_inf) {
        anomalies.push_back("-Inf");
      }
      if (is_positive_inf) {
        anomalies.push_back("+Inf");
      }
      if (is_nan) {
        anomalies.push_back("NaN");
      }
      string all_anomalies;
      if (anomalies.size() == 3) {
        all_anomalies = strings::StrCat(anomalies[0], ", ", anomalies[1],
                                        ", and ", anomalies[2]);
      } else if (anomalies.size() == 2) {
        all_anomalies = strings::StrCat(anomalies[0], " and ", anomalies[1]);
      } else {
        all_anomalies = anomalies[0];
      }
      context->SetStatus(errors::InvalidArgument(
          this->message_, " : Tensor had ", all_anomalies, " values"));
    }
  }

 private:
  string message_;
};

}  // namespace

REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    CheckNumericsOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("CheckNumerics")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::bfloat16>("T"),
                        CheckNumericsOp<GPUDevice, Eigen::bfloat16>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    CheckNumericsOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    CheckNumericsV2Op<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("CheckNumericsV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::bfloat16>("T"),
                        CheckNumericsV2Op<GPUDevice, Eigen::bfloat16>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    CheckNumericsV2Op<GPUDevice, float>);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    CheckNumericsOp<GPUDevice, double>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    CheckNumericsV2Op<GPUDevice, double>);
#endif  // ITEX_ENABLE_DOUBLE

}  // namespace itex
