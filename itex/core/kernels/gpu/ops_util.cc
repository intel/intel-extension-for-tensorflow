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

#include "itex/core/kernels/gpu/ops_util.h"

#include <algorithm>
#include <cmath>

#include "itex/core/utils/attr_value_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/padding.h"
#include "itex/core/utils/str_util.h"

namespace itex {

Eigen::PaddingType BrainPadding2EigenPadding(Padding padding) {
  switch (padding) {
    case Padding::VALID:
      return Eigen::PADDING_VALID;
    case Padding::SAME:
      return Eigen::PADDING_SAME;
    case Padding::EXPLICIT:
      ITEX_LOG(FATAL)
          << "Eigen does not have explicit padding enum "  // Crash OK
             "value";
  }
  return Eigen::PADDING_SAME;  // Prevent compiler warning about missing return
}

}  // namespace itex
