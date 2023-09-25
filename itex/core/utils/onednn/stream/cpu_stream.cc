#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/onednn/stream/mkl_threadpool.h"

namespace itex {
dnnl::stream CreateDnnlStream(const OpKernelContext& ctx,
                              const dnnl::engine& engine, int num_thread) {
  if (num_thread == 1) return dnnl::stream(engine);
  MklDnnThreadPool* eigen_tp = new MklDnnThreadPool(&ctx, num_thread);
  dnnl::stream tp_stream =
      dnnl::stream(dnnl::threadpool_interop::make_stream(engine, eigen_tp));
  return tp_stream;
}
}  // namespace itex
