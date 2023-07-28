# Python APIs

## Overview

Intel® Extension for TensorFlow* provides flexible Python APIs to configure settings for different types of application scenarios.

##### Prerequisite: `import intel_extension_for_tensorflow as itex`

* [*itex.ConfigProto*](#itexconfigproto): ProtocolMessage for XPU configuration under different types of backends and optimization options.
* [*itex.GraphOptions*](#itexgraphoptions): ProtocolMessage for graph configuration optimization options.
* [*itex.AutoMixedPrecisionOptions*](#itexautomixedprecisionoptions): ProtocolMessage for auto mixed precision optimization options.
* [*itex.ShardingConfig*](#itexshardingconfig): ProtocolMessage for XPUAutoShard optimization options.
* [*itex.DebugOptions*](#itexdebugoptions): ProtocolMessage for debug options.
* [*itex.set_config*](#itexset_config): Public API for setting ConfigProto.
* [*itex.get_config*](#itexget_config): Public API for getting ConfigProto.
* [*itex.ops*](#itex-operators): Public API for extended XPU operations.
* [*itex.experimental_ops_override*](#itex-ops-override): Public API for override TensorFlow operations with ITEX ones.
* [*itex.version*](#itex-version): Public API for Intel® Extension for TensorFlow* and components version information.

## Python APIs and Environment Variable Names

You can easily configure and tune Intel® Extension for TensorFlow* run models using Python APIs and environment variables. We recommend Python APIs.

##### Python APIs and preserved environment variable Names

| Python APIs        | Default value | Environment Variables                                        | Default value                                | Definition                                                   |
| ------------------ | ------------------ | ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| `itex.ConfigProto` |`OFF`<br>`ON`<br>`ON`<br/>`OFF`<br/>`OFF`<br/> |`ITEX_ONEDNN_GRAPH` <br>`ITEX_LAYOUT_OPT`<br>`ITEX_REMAPPER`<br>`ITEX_AUTO_MIXED_PRECISION`<br>`ITEX_SHARDING` | `0`<br>`1`*<br>`1`<br/>`0`<br/>`0`<br/>| Set configuration options for specific backend type (`CPU`/`GPU`) and graph optimization. <br/> *`ITEX_LAYOUT_OPT` default `ON` in Intel GPU (except Intel® Data Center GPU Max Series) and default `OFF` in Intel CPU by hardware attributes|
| `itex.experimental_ops_override` |`N/A` |`N/A`                                           | OFF                                        | Call this function to automatically override the operators with same name in TensorFlow by `itex.ops`. |

**Notes:**
1. The priority for setting values is as follows: Python APIs > Environment Variables > Default value.
2. If `pip install intel-extension-for-tensorflow[xpu]`, both GPU and CPU backends will be installed, the default backend will be selected by the platform device situation. If the platform with Intel GPU, the activate backend will be `GPU`, otherwise, `CPU`. If GPU backend was installed by `pip install intel-extension-for-tensorflow[gpu]`, the backend will be `GPU`. If CPU backend was installed by `pip install intel-extension-for-tensorflow[cpu]`, the backend is `CPU`.

## Intel® Extension for TensorFlow* Config Protocol
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
| `sharding` |Toggle sharding <br/><br/>Currently only supports Intel GPUs with multi-tiles. Override the environment variable `ITEX_SHARDING`. Set if XPUAutoShard is enabled to benefit from sharding input data/graph to maximize hardware usage.<br/>Enable or disable the XPUAutoShard. The default value is `OFF`.<br/>  <br/> * If `ON`, will enable XPUAutoShard optimization.<br/> * If `OFF`, will disable XPUAutoShard optimization.|

Examples:

I. Setting the options while creating the config protocol object
```python
# TensorFlow and Intel® Extension for TensorFlow*
import tensorflow as tf
import intel_extension_for_tensorflow as itex

graph_opts=itex.GraphOptions(onednn_graph=itex.ON)
config=itex.ConfigProto(graph_options=graph_opts)
print(config)
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

config=itex.ConfigProto()

config.graph_options.onednn_graph=itex.ON
config.graph_options.layout_opt=itex.OFF

print(config)
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

### itex.ShardingConfig

ProtocolMessage for XPUAutoShard optimization options. Currently only supports Intel GPUs with multi-tiles.

Refer to [XPUAutoShard on GPU](XPUAutoShard.md).

### itex.DebugOptions

ProtocolMessage for debug options.

| Python APIs                     | Environment Variables                | Definition                                                   |
| ------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| `auto_mixed_precision_log_path` | `ITEX_AUTO_MIXED_PRECISION_LOG_PATH` | Save auto mixed precision "pre-optimization" and "post-optimization" graph to log path. |
| `xpu_force_sync` | `ITEX_SYNC_EXEC` | Run the graph with sync mode. The default value is `OFF`. If `ON`, the whole model will be run with sync mode, which will hurt performance. |

### itex.set_config
Set Config Protocol. Note that the protocol is a global value, so this API is not thread safe.

```
itex.set_config(config)
```
| Args                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `config`      | [ConfigProto]((#ITEX-config-protocol)) object|

| Raises                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `ValueError`      | If argument validation fails.|


### itex\.get_config
Get Config Protocol.

```
itex.get_config()
```

| Raises                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `Returns`      | Return the current config.|

Example:

```python
import intel_extension_for_tensorflow as itex

graph_opts=itex.GraphOptions(onednn_graph=itex.ON)
config=itex.ConfigProto(graph_options=graph_opts)
itex.set_config(config)
print(itex.get_config())
```
Then the log will output the information like below:
```
graph_options {
  onednn_graph: ON
}
```

## itex operators

**itex.ops: Public API for extended XPU ops(operations) for itex.ops namespace.**

For details, refer to [Customized Operators](itex_ops.md).

## itex ops override

**itex.experimental_ops_override: Public API to override TensorFlow specific operators with same name by Customized Operators in itex.ops namespace.**

For details, refer to [Intel® Extension for TensorFlow* ops override](itex_ops_override.md).

Example:

```python
import intel_extension_for_tensorflow as itex
import tensorflow as tf
itex.experimental_ops_override()
print(tf.nn.gelu == itex.ops.gelu)
```
Then it will output the result "True".

## itex graph

**itex.graph: Public API for extended ITEX graph optimization operations.**

N/A

## itex version

**itex.version: Public API for itex.version namespace.**

| Other Members                   |                                     Description                         |
| -----------------------| ------------------------------------------------------------------------|
| `VERSION`      | The release version. For example, `0.3.0`|
| `GIT_VERSION`      | The git version. For example, `v0.3.0-7112d33`|
| `ONEDNN_CPU_GIT_VERSION`      | The oneDNN git version of CPU. For example, `v2.5.2-a930253`|
| `ONEDNN_GPU_GIT_VERSION`      | The oneDNN git version of GPU. For example, `v2.5.2-a930253`|
| `COMPILER_VERSION`      | The compiler version. For example, `gcc-8.2.1 20180905, dpcpp-2022.1.0.122`|
| `TF_COMPATIBLE_VERSION`      | The compatible TensorFlow versions. For example, `tensorflow >= 2.5.0, < 2.7.0, !=2.5.3, ~=2.6`|

Example:
```
import tensorflow as tf
import intel_extension_for_tensorflow as itex

print(itex.__version__)
print(itex.version.VERSION)
print(itex.version.GIT_VERSION)
if hasattr(itex.version, "ONEDNN_CPU_GIT_VERSION"):
  print(itex.version.ONEDNN_CPU_GIT_VERSION)    # For CPU or XPU
if hasattr(itex.version, "ONEDNN_GPU_GIT_VERSION"):
  print(itex.version.ONEDNN_GPU_GIT_VERSION)    # For GPU or XPU
print(itex.version.COMPILER_VERSION)
print(itex.version.TF_COMPATIBLE_VERSION)

```
