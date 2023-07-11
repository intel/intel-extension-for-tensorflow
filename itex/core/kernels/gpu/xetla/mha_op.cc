#include "itex/core/kernels/gpu/xetla/mha_op.h"

#include "itex/core/utils/errors.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {
#define REGISTER_MHA_GPU(type)                              \
  REGISTER_KERNEL_BUILDER(Name("ScaledDotProductAttention") \
                              .Device(DEVICE_GPU)           \
                              .TypeConstraint<type>("T"),   \
                          ScaledDotProductAttentionOp<GPUDevice, type>);

REGISTER_MHA_GPU(Eigen::bfloat16);
REGISTER_MHA_GPU(Eigen::half);
#undef REGISTER_MHA_GPU

#define REGISTER_MHA_GRAD_GPU(type)                             \
  REGISTER_KERNEL_BUILDER(Name("ScaledDotProductAttentionGrad") \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<type>("T"),       \
                          ScaledDotProductAttentionGradOp<GPUDevice, type>);

REGISTER_MHA_GRAD_GPU(Eigen::bfloat16);
REGISTER_MHA_GRAD_GPU(Eigen::half);
#undef REGISTER_MHA_GRAD_GPU
}  // namespace itex
