#include "itex/core/utils/onednn/onednn_util.h"

namespace itex {
dnnl::stream CreateDnnlStream(const OpKernelContext& ctx,
                              const dnnl::engine& engine, int num_thread) {
  ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::cpu)
      << "Create oneDNN stream for unsupported engine.";
  return dnnl::stream(engine);
}
}  // namespace itex
