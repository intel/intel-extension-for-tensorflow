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

#include "itex/core/kernels/common/transpose_functor.h"

#include "itex/core/utils/gtl/array_slice.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/status.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace internal {

#define INSTANTIATE(DEVICE)                                                 \
  template <>                                                               \
  Status DoTranspose(const DEVICE& device, const Tensor& in,                \
                     const gtl::ArraySlice<int32> perm, Tensor* out) {      \
    return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/false, \
                                     out);                                  \
  }                                                                         \
  template <>                                                               \
  Status DoConjugateTranspose(const DEVICE& device, const Tensor& in,       \
                              const gtl::ArraySlice<int32> perm,            \
                              Tensor* out) {                                \
    return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/true,  \
                                     out);                                  \
  }                                                                         \
  template <>                                                               \
  Status DoMatrixTranspose(const DEVICE& device, const Tensor& in,          \
                           Tensor* out) {                                   \
    return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/false, \
                                           out);                            \
  }                                                                         \
  template <>                                                               \
  Status DoConjugateMatrixTranspose(const DEVICE& device, const Tensor& in, \
                                    Tensor* out) {                          \
    return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/true,  \
                                           out);                            \
  }

template <typename Device, typename T>
void TransposeOnDevice(const Device& d, const Tensor& in,
                       const gtl::ArraySlice<int32> perm, bool conjugate,
                       Tensor* out) {
  switch (in.dims()) {
    case 2:
      TransposeUsingEigen<Device, T, 2>(d, in, perm, conjugate, out);
      break;
    case 3:
      TransposeUsingEigen<Device, T, 3>(d, in, perm, conjugate, out);
      break;
    case 4:
      TransposeUsingEigen<Device, T, 4>(d, in, perm, conjugate, out);
      break;
    case 5:
      TransposeUsingEigen<Device, T, 5>(d, in, perm, conjugate, out);
      break;
    case 6:
      TransposeUsingEigen<Device, T, 6>(d, in, perm, conjugate, out);
      break;
    case 7:
      TransposeUsingEigen<Device, T, 7>(d, in, perm, conjugate, out);
      break;
    case 8:
      TransposeUsingEigen<Device, T, 8>(d, in, perm, conjugate, out);
      break;
    default:
      ITEX_CHECK(false);
      break;
  }
}
}  // namespace internal

#define TRANSPOSE_INSTANTIATE(DEVICE)                                      \
  template <typename T, bool conjugate>                                    \
  struct Transpose<DEVICE, T, conjugate> {                                 \
    static void run(const DEVICE& d, const Tensor& in,                     \
                    const gtl::ArraySlice<int32> perm, Tensor* out) {      \
      internal::TransposeOnDevice<DEVICE, T>(d, in, perm, conjugate, out); \
    }                                                                      \
  };

#ifdef INTEL_CPU_ONLY
TRANSPOSE_INSTANTIATE(CPUDevice)
INSTANTIATE(CPUDevice)
#else
template <bool conjugate>
struct Transpose<GPUDevice, string, conjugate> {
  static void run(const GPUDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
    ITEX_LOG(FATAL) << "DT_STRING not supported on GPU device.";
  }
};

// Explicit instantiation.
template struct Transpose<GPUDevice, string, false>;

TRANSPOSE_INSTANTIATE(GPUDevice)
INSTANTIATE(GPUDevice)
#endif  // INTEL_CPU_ONLY

#undef TRANSPOSE_INSTANTIATE
#undef INSTANTIATE
}  // namespace itex
