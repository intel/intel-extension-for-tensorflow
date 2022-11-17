# oneDNN object cache optimization

## Introduction

The OneDNN object (primitve/primitive description/memory) creation overhead becomes noticable, especially in small model latency scenarios. 

OneDNN object cache optimization is experimental feature for optimizing model latency by binding an oneDNN object to a TensorFlow graph node. You can enable this environment by setting the environment variable 'ITEX_CACHE_ONEDNN_OBJECT' to on. By default, it is off.

TensorFlow supports optimizations to support different scenarios:

- **Dynamic Shape** - TensorFlow supports dynamic shape, which means a node may get different shape input. This optimization will invalid the cache by checking the input dims/shape with the oneDNN meta input (used in layout propagation).

- **Operator Parallel Execution** - TensorFlow supports [operator parallel execution](https://www.tensorflow.org/api_docs/python/tf/config/threading), which means a node may execute in different schedule threads. The oneDNN requires thread safe in this scenario only: **user scratchpad** and **oneDNN stream creation on demand**. This optimization is aligning to satisfy a oneDNN requirement.

- **Concurrently Execution** - Tensorflow supports [concurrently execution](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session#as_default), which means a node may be executed in different thread concurrently. This optimization handles this case by adding a mutex lock.

## Optimization in convolution

Convolution optimization will cache oneDNN object [dnnl::memory](https://oneapi-src.github.io/oneDNN/struct_dnnl_memory-2.html), [dnnl::primitive](https://oneapi-src.github.io/oneDNN/struct_dnnl_primitive-2.html), [dnnl::primitive_desc](https://oneapi-src.github.io/oneDNN/struct_dnnl_primitive_desc-2.html) and [dnnl_exec_arg_t](https://oneapi-src.github.io/oneDNN/struct_dnnl_exec_arg_t.html)

- **dnnl::memory** - input/weight/bias/output/scratchpad memory and two temporary memory for input and weight reorder if needed.
- **dnnl::primitive** - convolution primitive and input/weight reorder primitive if needed.
- **dnnl::primitive_desc** - convolution primitive description.
- **dnnl_exec_arg_t** - convolution primitive arguments and input/weight reorder primitive arguments if needed.

Temporary device memory includes scratchpad memory and input/weight reorder output device memory if needed.
