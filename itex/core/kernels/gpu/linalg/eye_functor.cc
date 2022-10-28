/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/kernels/gpu/linalg/eye_functor.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/register_types_traits.h"
#include "itex/core/utils/tensor_types.h"

namespace itex {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct Eye {
  Eye(int32_t m, int32_t n, T* matrix) : m(m), n(n), matrix(matrix) {}
  void operator()(sycl::item<1> item) const {
    const int index = item.get_id();
    const int global_row = index / n;
    const int col = index - global_row * n;
    const int batch = global_row / m;
    const int row = global_row - batch * m;
    matrix[index] = col == row ? T(1) : T(0);
  }

 private:
  int32_t m;
  int32_t n;
  T* matrix;
};

template <typename Scalar>
void LaunchEyeKernel(sycl::queue* stream, int batch_size, int m, int n,
                     Scalar* matrix) {
  stream->submit([&](sycl::handler& cgh) {
    Eye<Scalar> task(m, n, matrix);
    cgh.parallel_for<Eye<Scalar>>(sycl::range<1>(batch_size * m * n), task);
  });
}

template <typename Scalar>
struct EyeFunctor<GPUDevice, Scalar> {
  void operator()(sycl::queue* stream,
                  typename TTypes<Scalar, 3>::Tensor matrix_batch) {
    const int batch_size = matrix_batch.dimension(0);
    const int m = matrix_batch.dimension(1);
    const int n = matrix_batch.dimension(2);
    LaunchEyeKernel(stream, batch_size, m, n, matrix_batch.data());
  }
};

template struct EyeFunctor<GPUDevice, float>;
#ifdef ITEX_ENABLE_DOUBLE
template struct EyeFunctor<GPUDevice, double>;
template struct EyeFunctor<GPUDevice, complex128>;
#endif  // ITEX_ENABLE_DOUBLE
template struct EyeFunctor<GPUDevice, complex64>;
}  // namespace functor
}  // namespace itex
