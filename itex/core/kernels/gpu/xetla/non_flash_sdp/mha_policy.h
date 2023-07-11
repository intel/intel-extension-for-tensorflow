#ifndef ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_POLICY_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_POLICY_H_

#include <xetla.hpp>

namespace gpu::xetla {

struct mha_policy_base {
  static constexpr uint32_t accum_step = 32;
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t sync_freq = 0;
};

/*
Note:
  kHm / kSgHm == kTm / kSgTm
  kSgHm and kSgTm should be a multiple of 16
  kSgBr should be a multiple of 8
*/

struct mha_policy_64x384x64 : mha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kTm = 384;
  static constexpr uint32_t kSgTm = 64;
  static constexpr uint32_t kHm = 192;
  static constexpr uint32_t kSgHm = 32;
};

struct mha_policy_32x512x128 : mha_policy_base {
  static constexpr uint32_t kBr = 32;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kTm = 512;
  static constexpr uint32_t kSgTm = 64;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 16;
};

}  // namespace gpu::xetla

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_NON_FLASH_SDP_MHA_POLICY_H_
