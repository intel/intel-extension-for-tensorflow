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

#ifndef ITEX_CORE_UTILS_TF_VERSION_H_
#define ITEX_CORE_UTILS_TF_VERSION_H_

#ifndef ITEX_BUILD_JAX
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/c_api.h"

namespace itex {
class TensorFlowVersion {
 public:
  TensorFlowVersion() {
    absl::string_view tf_ver = TF_Version();
    decode_str(tf_ver);
  }

  explicit TensorFlowVersion(absl::string_view tf_ver) { decode_str(tf_ver); }

  void decode_str(absl::string_view tf_ver) {
    std::vector<std::string> ver_split = absl::StrSplit(tf_ver, '.');
    major_ = std::stoi(ver_split[0]);
    minor_ = std::stoi(ver_split[1]);

    std::vector<std::string> ver_last_split = absl::StrSplit(ver_split[2], '-');
    patch_ = std::stoi(ver_last_split[0]);
    if (ver_last_split.size() > 1) {
      version_suffix_ = ver_last_split[1];
    } else {
      version_suffix_ = "";
    }
  }

  // TODO(itex): Consider version suffix in comparison
  friend bool operator==(const TensorFlowVersion& a,
                         const TensorFlowVersion& b) {
    return a.major_ == b.major_ && a.minor_ == b.minor_ && a.patch_ == b.patch_;
  }

  friend bool operator!=(const TensorFlowVersion& a,
                         const TensorFlowVersion& b) {
    return !(a == b);
  }

  friend bool operator<(const TensorFlowVersion& a,
                        const TensorFlowVersion& b) {
    return a.major_ < b.major_ || a.minor_ < b.minor_ || a.patch_ < b.patch_;
  }

  friend bool operator>(const TensorFlowVersion& a,
                        const TensorFlowVersion& b) {
    return b < a;
  }

  friend bool operator<=(const TensorFlowVersion& a,
                         const TensorFlowVersion& b) {
    return !(a > b);
  }

  friend bool operator>=(const TensorFlowVersion& a,
                         const TensorFlowVersion& b) {
    return !(a < b);
  }

  friend bool operator==(const TensorFlowVersion& a, absl::string_view b) {
    return a == TensorFlowVersion(b);
  }

  friend bool operator!=(const TensorFlowVersion& a, absl::string_view b) {
    return a != TensorFlowVersion(b);
  }

  friend bool operator<(const TensorFlowVersion& a, absl::string_view b) {
    return a < TensorFlowVersion(b);
  }

  friend bool operator>(const TensorFlowVersion& a, absl::string_view b) {
    return a > TensorFlowVersion(b);
  }

  friend bool operator<=(const TensorFlowVersion& a, absl::string_view b) {
    return a <= TensorFlowVersion(b);
  }

  friend bool operator>=(const TensorFlowVersion& a, absl::string_view b) {
    return a >= TensorFlowVersion(b);
  }

  std::string ToString() const {
    return std::to_string(major_) + "." + std::to_string(minor_) + "." +
           std::to_string(patch_) + version_suffix_;
  }

 private:
  int major_;
  int minor_;
  int patch_;
  std::string version_suffix_;
};

std::ostream& operator<<(std::ostream& os, const TensorFlowVersion& x) {
  os << x.ToString();
  return os;
}

}  // namespace itex

#endif
#endif  // ITEX_CORE_UTILS_TF_VERSION_H_
