# Python APIs

## Overview

Intel® Extension for TensorFlow* provides flexible Python APIs to configure settings for different types of application scenarios.

##### Prerequisite: `import intel_extension_for_tensorflow as itex`

* [*itex.set_backend*](#set-itex-backend): Public API for setting backend type and options.
* [*itex.get_backend*](#set-itex-backend): Public API for getting backend type.
* [*itex.ConfigProto*](#ITEX-config-protocol): ProtocolMessage for XPU configuration under different types of backends and optimization options.
* [*itex.GPUOptions*](#ITEX-config-protocol): ProtocolMessage for GPU configuration optimization options.
* [*itex.GraphOptions*](#ITEX-config-protocol): ProtocolMessage for graph configuration optimization options.
* [*itex.AutoMixedPrecisionOptions*](#ITEX-config-protocol): ProtocolMessage for auto mixed precision optimization options.
* [*itex.DebugOptions*](#ITEX-config-protocol): ProtocolMessage for debug options.
* [*itex.ops*](#itex-ops): Public API for extended XPU operations.
* [*itex.version*](#itex-version): Public API for Intel® Extension for TensorFlow* and components version information.

## Python APIs and Environment Variable Names

You can easily configure and tune Intel® Extension for TensorFlow* run models using Python APIs and environment variables. We recommend Python APIs.

##### Python APIs and preserved environment variable Names

| Python APIs        | Default value | Environment Variables                                        | Default value                                | Definition                                                   |
| ------------------ | ------------------ | ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| `itex.set_backend` |`GPU`or`CPU` |`ITEX_XPU_BACKEND`                                           | `GPU`or`CPU`                                        | set `CPU`/`GPU` as specific `XPU` backend with optimization options for execution.  |
| `itex.get_backend` |`N/A`| `N/A`                                                        | `N/A`                                        | Get the string of current XPU backend. For example `CPU`, `GPU` or `AUTO`. |
| `itex.ConfigProto` |`OFF`<br>`ON`<br>`ON`<br/>`OFF`<br/> |`ITEX_ONEDNN_GRAPH` <br>`ITEX_LAYOUT_OPT`<br>`ITEX_REMAPPER`<br>`ITEX_AUTO_MIXED_PRECISION` | `0`<br>`1`*<br>`1`<br/>`0`<br/>| Set configuration options for specific backend type (`CPU`/`GPU`) and graph optimization. <br/> *`ITEX_LAYOUT_OPT` default `ON` in Intel GPU (except Ponte Vecchio) and default `OFF` in Intel CPU by hardware attributes|

**Notes:**
1. The priority for setting values is as follows: Python APIs > Environment Variables > Default value.
2. If GPU backend was installed by `pip install intel-extension-for-tensorflow[gpu]`, the default backend will be `GPU`. If CPU backend was installed by `pip install intel-extension-for-tensorflow[cpu]`, the default backend is `CPU`.

## Set Intel® Extension for TensorFlow* Backend

### itex.set_backend
Intel® Extension for TensorFlow* provides multiple types of backends with different optimization options to execute. Only one backend is allowed in the whole process, and this can only be configured once before XPU device initialization.

Set `CPU`/`GPU` with `config` as specific XPU backend type for execution.

```
itex.set_backend (
  backend='GPU',
  config=itex.ConfigProto()
)
```

| Args                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `backend`      | The backend type to set. The default value is `CPU`.<br>  <br> * If `GPU`, the XPU backend type is set as `GPU` and all ops will be executed on concrete GPU backend.<br> * If `CPU`, the XPU backend type is set as `CPU` and all ops will be executed on concrete CPU backend. <br><br> * If CPU backend was installed by `pip install intel-extension-for-tensorflow[cpu]`, it's invalid to set XPU backend type as `GPU`.|
| `config`| The backend config to set. Refer to [itex.ConfigProto](#itex-config-protocol) for details.|

| Raises                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `ValueError`      | If argument validation fails, or the backend type cannot be changed.|
| `RuntimeWarning`      | This API is called after XPU device initialization or called more than one time.|


Examples:

I. Set the specific XPU backend type and config for `tf.device("/xpu:0")`.

```python
# TensorFlow graph mode or eager mode
import tensorflow as tf
import intel_extension_for_tensorflow as itex

backend_cfg=itex.ConfigProto(gpu_options=itex.GPUOptions(onednn_graph=itex.ON))

# Only allow this setting once in backend device initialization
# All operators will be executed in `GPU` backend with the option setting.
itex.set_backend('GPU', backend_cfg)

def add_func(x, y):
    return x+y

with tf.device("/xpu:0"):
    print(add_func(1, 1))
```

II. Set the specific XPU backend type and config for a device not explicitly specified.

```python
# TensorFlow graph mode or eager mode
import tensorflow as tf
import intel_extension_for_tensorflow as itex

#Only allow setting once in backend device initialization
#All operators will be executed in `GPU` backend with two options setting.
backend_cfg=itex.ConfigProto()
backend_cfg.graph_options.onednn_graph=itex.OFF
backend_cfg.graph_options.layout_opt=itex.ON

itex.set_backend('GPU', backend_cfg)

def add_func(x, y):
    return x+y

print(add_func(1, 1))
```

### itex\.get_backend
Get the string of current XPU backend type. For example `CPU` or `GPU`.

```
itex.get_backend ()
```

| Raises                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `Returns`      | Return the current XPU backend type string.|

The following example demonstrates setting the XPU backend type as `GPU` and checking its value on the machine, while GPU backend is installed by `pip install intel-extension-for-tensorflow[gpu]`.

```
# TensorFlow and Intel® Extension for TensorFlow*
import tensorflow as tf
import intel_extension_for_tensorflow as itex

# Only allow setting once in backend device initialization
itex.set_backend('GPU')

print(itex.get_backend())
```
Then the log will output `GPU`.

## Itex Config Protocol
**itex.ConfigProto: ProtocolMessage for XPU configuration under different types of backends and optimization options.**

**enum class**

| enum class                                                   | Descriptions                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| enum ITEXDataType {<br/>  DEFAULT_DATA_TYPE = 0;<br/>  FLOAT16 = 1;<br/>  BFLOAT16 = 2;<br/>} | Datatype options of advanced auto mixed precision. You could set datatype for advanced auto mixed precision on CPUs or GPUs. |
| enum Toggle {<br/>    DEFAULT = 0;<br/>    ON = 1;<br/>    OFF = 2;<br/>} | Configuration options for the graph optimizer. Unless otherwise noted, these configuration options do not apply to explicitly triggered optimization passes in the optimizers field. |

**Functions**

### itex.ConfigProto

| Attribute                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `gpu_options`      | GPUOptions protocolMessage, `GPU` backend options.|
| `cpu_options`      | CPUOptions protocolMessage, `CPU` backend options.|
| `auto_options`     | XPUOptions protocolMessage, `XPU` backend options.|
| `graph_options`    | GraphOptions protocolMessage, graph optimization options.|

### itex.GPUOptions

| Attribute                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `None`     | N/A |

### itex.GraphOptions

| Attribute                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `onednn_graph` |Toggle onednn_graph<br><br>Override the environment variable `ITEX_ONEDNN_GRAPH`. Set to enable or disable oneDNN graph(LLGA) optimization. The default value is `OFF`.<br>  <br> * If `ON`, will enable oneDNN graph in Intel® Extension for TensorFlow*.<br> * If `OFF`, will disable oneDNN graph in Intel® Extension for TensorFlow*.|
| `layout_opt ` |Toggle layout_opt <br><br>Override the environment variable `ITEX_LAYOUT_OPT`. Set if oneDNN layout optimization is enabled to benefit from oneDNN block format.<br> Enable or disable the oneDNN layout. The default value is `OFF`.<br>  <br> * If `ON`, will enable oneDNN layout optimization.<br> * If `OFF`, will disable oneDNN layout optimization.|
| `remapper` |Toggle remapper <br/><br/>Override the environment variable `ITEX_REMAPPER`. Set if remapper optimization is enabled to benefit from sub-graph fusion.<br/> Enable or disable the remapper. The default value is `ON`.<br/>  <br/> * If `ON`, will enable remapper optimization.<br/> * If `OFF`, will disable remapper optimization.|
| `auto_mixed_precision` |Toggle auto_mixed_precision <br/><br/>Override the environment variable `ITEX_AUTO_MIXED_PRECISION`. Set if mixed precision is enabled to benefit from using both 16-bit and 32-bit floating-point types to accelerate modes.<br/>Enable or disable the  auto mixed precision. The default value is `OFF`.<br/>  <br/> * If `ON`, will enable auto mixed precision optimization.<br/> * If `OFF`, will disable auto mixed precision optimization.|

Examples:

I. Setting the options while creating the config protocol object
```python
# TensorFlow and Intel® Extension for TensorFlow*
import tensorflow as tf
import intel_extension_for_tensorflow as itex

graph_opts=itex.GraphOptions(onednn_graph=itex.ON)
backend_cfg=itex.ConfigProto(graph_options=graph_opts)
print(backend_cfg)
```
Then the log will output the information like below.
```python
graph_options {
  onednn_graph: ON
}
```
II. Setting the options after creating the config protocol object

```
# TensorFlow and Intel® Extension for TensorFlow*
import tensorflow as tf
import intel_extension_for_tensorflow as itex

backend_cfg=itex.ConfigProto()

backend_cfg.graph_options.onednn_graph=itex.ON
backend_cfg.graph_options.layout_opt=itex.OFF

print(backend_cfg)
```
Then the log will output the information like below.
```
graph_options {
  onednn_graph: ON
  layout_opt: OFF
}
```

### itex.AutoMixedPrecisionOptions

ProtocolMessage for auto mixed precision optimization options.

Refer to [Advanced Auto Mixed Precision](advanced_auto_mixed_precision.md).

### itex.DebugOptions

ProtocolMessage for debug options.

| Python APIs                     | Environment Variables                | Definition                                                   |
| ------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| `auto_mixed_precision_log_path` | `ITEX_AUTO_MIXED_PRECISION_LOG_PATH` | Save auto mixed precision "pre-optimization" and "post-optimization" graph to log path. |
| `xpu_force_sync` | `ITEX_SYNC_EXEC` | Run the graph with sync mode. The default value is `OFF`. If `ON`, the whole model will be run with sync mode, which will hurt performance. |

## itex operators

**itex.ops: Public API for extended XPU ops(operations) for itex.ops namespace.**

For details, refer to [ITEX ops](itex_ops.md).


## itex graph

**itex.graph: Public API for extended ITEX graph optimization operations.**

N/A

## itex version

**itex.version: Public API for itex.version namespace.**

| Other Members                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `VERSION`      | The release version. For example, `0.3.0`|
| `GIT_VERSION`      | The git version. For example, `v0.3.0-7112d33`|
| `ONEDNN_GIT_VERSION`      | The oneDNN git version. For example, `v2.5.2-a930253`|
| `COMPILER_VERSION`      | The compiler version. For example, `gcc-8.2.1 20180905, dpcpp-2022.1.0.122`|
| `TF_COMPATIBLE_VERSION`      | The compatible TensorFlow versions. For example, `tensorflow >= 2.5.0, < 2.7.0, !=2.5.3, ~=2.6`|

Example:
```
import tensorflow as tf
import intel_extension_for_tensorflow as itex

print(itex.__version__)
print(itex.version.VERSION)
print(itex.version.GIT_VERSION)
print(itex.version.ONEDNN_GIT_VERSION)
print(itex.version.COMPILER_VERSION)
print(itex.version.TF_COMPATIBLE_VERSION)

```
