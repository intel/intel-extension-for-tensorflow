/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/multinomial_op.h"
#include "itex/core/kernels/gpu/random_op_gpu.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"

namespace itex {

using GPUDevice = Eigen::GpuDevice;

namespace impl {

template <typename T, typename OutputType>
struct MultinomialKernel {
  MultinomialKernel(int batch_size, int num_samples, int num_classes,
                    const float* scores, OutputType* output)
      : batch_size(batch_size),
        num_samples(num_samples),
        num_classes(num_classes),
        scores(scores),
        output(output) {}
  void operator()(sycl::nd_item<1> item) const {
    auto output_id = item.get_global_linear_id();
    if (output_id >= batch_size * num_samples) {
      return;
    }
    auto batch_id = output_id / num_samples;
    auto sample_id = output_id % num_samples;

    auto offset =
        batch_id * (num_samples * num_classes) + sample_id * num_classes;
    float max_so_far = -1 * std::numeric_limits<float>::max();

    for (int i = 0; i < num_classes; i++) {
      max_so_far = sycl::fmax(max_so_far, scores[offset + i]);
    }

    for (int i = 0; i < num_classes; i++) {
      if (max_so_far == scores[offset + i]) {
        output[output_id] = static_cast<OutputType>(i);
      }
    }
  }

 private:
  int batch_size;
  int num_samples;
  int num_classes;
  const float* scores;
  OutputType* output;
};

template <typename T, typename OutputType>
Status LaunchMultinomialKernel(const GPUDevice& d, const float* scores,
                               int batch_size, int num_samples, int num_classes,
                               OutputType* output) {
  auto stream = d.stream();
  const auto work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const auto num_workgroup =
      (batch_size * num_samples + work_group_size - 1) / work_group_size;

  stream->submit([&](sycl::handler& cgh) {
    MultinomialKernel<T, OutputType> task(batch_size, num_samples, num_classes,
                                          scores, output);
    cgh.parallel_for<MultinomialKernel<T, OutputType>>(
        sycl::nd_range<1>(sycl::range<1>(num_workgroup * work_group_size),
                          sycl::range<1>(work_group_size)),
        task);
  });
  return Status::OK();
}
}  // end namespace impl

namespace functor {
// Kernel for Multinomial op.  Data is interpreted to have the following shapes:
// scores: [B, S, C];  maxima: [B, S];  output: [B, S].
template <typename T, typename OutputType>
struct MultinomialFunctor<GPUDevice, T, OutputType> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat noises,
                  typename TTypes<float>::Flat scores,
                  typename TTypes<float>::Flat /* scratch */, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<OutputType>::Matrix output) {
    // Uniform, [0, 1).
    typedef random::UniformDistribution<random::PhiloxRandom, float> Dist;
    functor::FillPhiloxRandom<GPUDevice, Dist>()(ctx, d, gen, noises.data(),
                                                 noises.size(), Dist());
#if defined(EIGEN_HAS_INDEX_LIST)
    Eigen::IndexList<int, Eigen::type2index<1>, int> boc;
    boc.set(0, batch_size);
    boc.set(2, num_classes);

    Eigen::IndexList<Eigen::type2index<1>, int, Eigen::type2index<1>> oso;
    oso.set(1, num_samples);
#else
    Eigen::array<int, 3> boc{batch_size, 1, num_classes};
    Eigen::array<int, 3> oso{1, num_samples, 1};
#endif

    // Calculates "scores = logits - log(-log(noises))"; B*C*S elements.
    // NOTE: we don't store back to "noises" because having it appear on both
    // sides is potentially unsafe (e.g. Eigen may use ldg() to load RHS data).
    // 2e-30 is chosen so as to be small enough to only change 0 -> 2e-30 while
    // not affect any of the other numbers (smallest is ~1e-7), but not so small
    // that log(x) == -inf, which is why it needs to be larger than 0 in the
    // first place.
    To32Bit(scores).device(d) =
        To32Bit(logits).reshape(boc).broadcast(oso).template cast<float>() -
        ((-((To32Bit(noises) + 2e-30f).log())).log());

    auto status = impl::LaunchMultinomialKernel<T, OutputType>(
        d, scores.data(), batch_size, num_samples, num_classes, output.data());

    OP_REQUIRES_OK(ctx, status);

    return;
  }
};

// Explicit instantiation of the GPU functors.
template struct MultinomialFunctor<GPUDevice, Eigen::half, int32>;
template struct MultinomialFunctor<GPUDevice, Eigen::bfloat16, int32>;
template struct MultinomialFunctor<GPUDevice, float, int32>;
template struct MultinomialFunctor<GPUDevice, int32, int32>;
template struct MultinomialFunctor<GPUDevice, int64, int32>;

template struct MultinomialFunctor<GPUDevice, Eigen::half, int64>;
template struct MultinomialFunctor<GPUDevice, Eigen::bfloat16, int64>;
template struct MultinomialFunctor<GPUDevice, float, int64>;
template struct MultinomialFunctor<GPUDevice, int32, int64>;
template struct MultinomialFunctor<GPUDevice, int64, int64>;

#ifdef ITEX_ENABLE_DOUBLE
template struct MultinomialFunctor<GPUDevice, double, int32>;
template struct MultinomialFunctor<GPUDevice, double, int64>;
#endif  // ITEX_ENABLE_DOUBLE

}  // namespace functor
}  // namespace itex
