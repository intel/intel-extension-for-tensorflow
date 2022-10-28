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

#include "itex/core/kernels/gpu/random_op_gpu.h"
#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::UniformDistribution<random::PhiloxRandom, Eigen::bfloat16> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int32> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int64> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, float> >;

template struct FillPhiloxRandom<
    GPUDevice,
    random::NormalDistribution<random::PhiloxRandom, Eigen::bfloat16> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, float> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::bfloat16> >;

#ifdef ITEX_ENABLE_DOUBLE
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, double> >;
#endif  // ITEX_ENABLE_DOUBLE
}  // namespace functor
}  // namespace itex
