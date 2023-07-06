
# How to write custom op

## 1.  Prerequisite
* Before code changes, please make sure the environment setting and source code build pass by  [build procedure](../install/how_to_build.md).
* Check TensorFlow version.
    ```bash
    $ python -c "import tensorflow as tf;print(tf.__version__)
    ```
* Check Intel Extension for TensorFlow* with verbose.
    ```bash
    $ export ITEX_VERBOSE=1
    $ export ONEDNN_VERBOSE=1
    ```
    Refer to [quick example](../../examples/quick_example.md).    
    Refer to [Intel® Extension for Tensorflow* Code Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-extension-for-tensorflow-code.html) to familiar with source code architecture. Custom op is built into all Intel® Extension for Tensorflow* library.    
    Refer to [TensorFlow Guide for Creating OP](https://www.tensorflow.org/guide/create_op) for TensorFlow offcial doc.

## 2.  Define the op interface and Register op
Take **GeluOp** as an example.
> [*itex/core/ops/op_init.h*](../../itex/core/ops/op_init.cc)
> [*itex/core/ops/op_init.cc* ](../../itex/core/ops/op_init.cc)    
```python   
void  Register_GeluOp();
```
Declare and call `Register_GeluOp()`
> [*itex/core/ops/nn_ops.cc*](../../itex/core/ops/nn_ops.cc)

```python
void Register_GeluOp() {
  ITEX_VLOG(0) << "#### Register_GeluOp";
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("Gelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "approximate: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Gelu op registration failed: ";
  }
}
```
Specify the name of the op, its inputs (types and names) and outputs (types and names), as well as docstrings and any attrs the op might require.
* Note：You can add `ITEX_VLOG(0) << "#### Register_GeluOp";` in the code begining for debug.



## 3.  Register the kernels for the op
For example, **GeluOp**, one kernel made for CPUs, and a separate one for GPUs.
> [*itex/core/kernels/cpu/relu_op.cc*](../../itex/core/kernels/cpu/relu_op.cc) Register **CPU** kernel **Gelu** for **GeluOp**
```python
REGISTER_KERNEL_BUILDER(                                   \
Name("Gelu").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
GeluOp<CPUDevice, type>);                                  \
```

> [*itex/core/kernels/gpu/relu_op.cc*](../../itex/core/kernels/gpu/relu_op.cc) Register **GPU** kernel **Gelu** for **GeluOp**
```python
REGISTER_KERNEL_BUILDER(                                   \
Name("Gelu").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
GeluOp<GPUDevice, type>);                                  \
```

**Device** determines the engine type, cpu or gpu   
**"T"** makes the kernel type polymorphism.   

Note: `load_ops_library` will load the library, and ops/kernels registered in the library via the`REGISTER_*` macros are made available in the TensorFlow process. For example, we can use `load_ops_library.gelu()` in python directly.    
> [*itex/python/ops/load_ops_library.py*](../../itex/python/ops/load_ops_library.py)   


## 4.  Implement the kernels

For example, GeluOp construction,  `GeluOp -> ReluBaseOp -> EltwiseBaseOp -> OpKernel`
> [*itex/core/kernels/common/relu_op.h*](../../itex/core/kernels/common/relu_op.h).  

```C++
template <typename Device, typename T>
class GeluOp : public ReluBaseOp<Device, T> {
 public:
  explicit GeluOp(OpKernelConstruction* context)
      : ReluBaseOp<Device, T>(context, dnnl::algorithm::eltwise_gelu_erf, 0.0f,
                              0.0f) {
    ITEX_VLOG(0) << "#### GeluOp construct";
    if (context->HasAttr("approximate")) {
      OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate_));
      this->alg_kind_ = approximate_ ? algorithm::eltwise_gelu_tanh
                                     : algorithm::eltwise_gelu_erf;
    }
  }
```
* Note: You can add `ITEX_VLOG(0)  <<  "#### GeluOp construct";` in the code for debug.
```C++
template <typename Device, typename T>
class ReluBaseOp : public EltwiseBaseOp<Device, T> {
 public:
  explicit ReluBaseOp(OpKernelConstruction* context, dnnl::algorithm algo,
                      float alpha, float beta)
      : EltwiseBaseOp<Device, T>(context, algo, alpha, beta) {}
};
```

> [*itex/core/kernels/common/eltwise_base.h*](../../itex/core/kernels/common/eltwise_base.h)  

Rewrite `EltwiseBaseOp compute`
```C++
template <typename Device, typename T>
class EltwiseBaseOp : public OpKernel {
 public:
  explicit EltwiseBaseOp(OpKernelConstruction* ctx, dnnl::algorithm algo,
                         float alpha, float beta)
      : OpKernel(ctx), alg_kind_(algo), alpha_(alpha), beta_(beta) {}

  void Compute(OpKernelContext* context) override {
    ITEX_VLOG(0) << "#### EltwiseBaseOp compute";
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      ......
      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_tensor.shape().num_elements() == 0) {
        OP_REQUIRES_OK(context,
                       context->allocate_output(kDstIndex, src_tensor.shape(),
                                                &dst_tensor));
        return;
      }
      ......
      // Create an eltwise forward descriptor and primitive descriptor
      eltwise_forward::desc fwd_desc(prop_kind::forward, alg_kind_, src_md,
                                     alpha_, beta_);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      eltwise_forward::primitive_desc fwd_pd(fwd_desc, attr, onednn_engine);
      ......
      primitive fwd_primitive(fwd_pd);
      ......
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> fwd_primitive_args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_DST, dst_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
      fwd_primitive.execute(onednn_stream, fwd_primitive_args);
    } catch (dnnl::error& e) {
      ......
    }
  }
 protected:
 algorithm alg_kind_ = algorithm::eltwise_relu;
 ......
};
```
* Notes:    
   * oneDNN gets the engine type **cpu** or **gpu** from `Device`.
   * `algorithm::eltwise_relu` is from oneDNN `dnnl.hpp`.  It finds eligible implementation from the list based on algorithm, engine, inference/forward/backward type, ... 
   * Then, create primitive descible, create primitive, execute primitive.    
     More oneDNN reference [click here](https://oneapi-src.github.io/oneDNN/page_getting_started_cpp.html#getting-started).
   * You can add `ITEX_VLOG(0) << "#### EltwiseBaseOp compute";` in the code for debug.


## 6.  Add the op to BUILD
Add C++/Header code to BUILD. Take **GeluOp** as an example for GPU.
> [[*itex/core/kernels/gpu/BUILD*]](../../itex/core/kernels/gpu/BUILD)
```python
itex_xpu_library(
    name = "relu_op",
    srcs = ["relu_op.cc"],
    hdrs = [
        "relu_op.h",
        "relu_op_functor.h",
        "//itex/core/kernels/common:eltwise_base_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)
...
GPU_KERNELS = [
...
":relu_op",
...
]
```
In [*itex/core/ops/BUILD*](../../itex/core/ops/BUILD), it adds **nn_ops** to **cc_library**.  
* Build Tips: （Optional）
Removing some kernels in [itex/core/kernels/gpu/BUILD](../../itex/core/kernels/gpu/BUILD) for GPU or  [itex/core/kernels/cpu/BUILD](../../itex/core/kernels/cpu/BUILD) for CPU can save compile time.

## 7. Use the op in Python
Take **GeluOp** as an example.
> [*itex/python/ops/activations.py*](../../itex/python/ops/activations.py) 
```python
with ops.name_scope(name, "Gelu", [features]):
    features = ops.convert_to_tensor(features, name="features")
    return load_ops_library.gelu(features, approximate)
```
> [*itex/python/base_init.py*](.././itex/python/base_init.py)
```python
  from intel_extension_for_tensorflow.python import ops
```
> [*itex/python/ops/__init__.py*](../../itex/python/ops/__init__.py)
```python
  from intel_extension_for_tensorflow.python.ops.activations import gelu
```

## 8. Build the package
```sh
$ bazel clean
$ git clean -xfd
$ ./configure
$ bazel build -c opt --config=gpu //itex/tools/pip_package:build_pip_package
$ bazel-bin/itex/tools/pip_package/build_pip_package ./
```
Refer to [here](../install/how_to_build.md) for more details.

## 9.  Install and Verify
```sh
$ pip uninstall intel_extension_for_tensorflow_lib
$ pip uninstall intel_extension_for_tensorflow
$ pip install intel_extension_for_tensorflow-*.whl
$ pip install intel_extension_for_tensorflow_lib-*.whl
```
For example:
```python
    import tensorflow as tf
    import intel_extension_for_tensorflow as itex
    x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
    y = itex.ops.gelu(x)
    print(y)
```
Run on GPU and print log:
```
I itex/core/ops/nn_ops.cc:141] #### Register_GeluOp
......
I tensorflow/core/common_runtime/eager/execute.cc:1445] Executing op Gelu in device /job:localhost/replica:0/task:0/device:XPU:0
I ./itex/core/kernels/common/relu_op.h:100] #### GeluOp construct
I ./itex/core/kernels/common/eltwise_base.h:44] #### EltwiseBaseOp compute
onednn_verbose,exec,gpu:0,eltwise,ocl:gen9:any,forward_training,data_f32::blocked:a:f0 diff_undef::undef::f0,attr-scratchpad:user ,alg:eltwise_gelu_erf alpha:0 beta:0,5,xxxxxx
I ./itex/core/utils/op_kernel.h:773] Gelu,Gelu,xxxxxx
```
