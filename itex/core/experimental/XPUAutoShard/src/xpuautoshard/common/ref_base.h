/* Copyright (c) 2023 Intel Corporation

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

#pragma once
#include <memory>
#include <utility>

namespace as {

/**
 * @brief A shared reference to the `Cls` type object
 *
 * @tparam Cls
 */
template <typename Cls>
using Ref = std::shared_ptr<Cls>;

/**
 * @brief Create an object of type `Cls` and return a shared reference to
 * its `BaseCls`.
 *
 * @tparam Cls
 * @tparam BaseCls
 * @tparam Args
 * @param args
 * @return Ref<BaseCls>
 */
template <typename Cls, typename BaseCls, typename... Args>
Ref<BaseCls> makeRef(Args&&... args) {
  return Ref<BaseCls>(new Cls(std::forward<Args>(args)...));
}

/**
 * @brief Create an object of type `Cls` and return a shared reference to it.
 *
 * @tparam Cls
 * @tparam Args
 * @param args
 * @return Ref<Cls>
 */
template <typename Cls, typename... Args>
Ref<Cls> makeRef(Args&&... args) {
  return Ref<Cls>(new Cls(std::forward<Args>(args)...));
}

/**
 * @brief Downcast a reference of `BaseCls` to its derative `Cls`.
 *
 * @tparam Cls
 * @tparam BaseCls
 * @param base
 * @return Ref<Cls>
 */
template <typename Cls, typename BaseCls>
Ref<Cls> downcastRef(Ref<BaseCls> base) {
  return std::dynamic_pointer_cast<Cls>(base);
}

/**
 * @brief Check if the given `cls` is of type `Cls`.
 *
 * @tparam Cls
 * @tparam SomeCls
 * @param cls
 * @return true
 * @return false
 */
template <typename Cls, typename SomeCls>
bool isRef(Ref<SomeCls> cls) {
  return std::dynamic_pointer_cast<Cls>(cls) != nullptr;
}

}  // namespace as
