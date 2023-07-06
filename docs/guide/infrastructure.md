Infrastructure
===============

## Architecture

<a target="_blank" href="images/architecture.png">
  <img src="images/architecture.png" alt="Architecture">
</a>

## Introduction

TensorFlow Public API
-----------------------------------------------------------

Intel® Extension for TensorFlow* is compatible with the stock TensorFlow public API definition, maintaining the same user experience of stock TensorFlow public API.


Custom API
-----------------------------------------------------------

Intel® Extension for TensorFlow* provides additional custom APIs to extend the stock TensorFlow public API. Both Python API and low level XPU kernels are implemented. Users can import and use the custom API under itex.ops namespace. Refer to [Customized Operators](itex_ops.md) for details.

Intel Advanced Feature and Extension Management
-----------------------------------------------------------

Generally, the default configuration of Intel® Extension for TensorFlow\* can provide good performance without any code changes. For advanced users, simple frontend Python APIs and utilities can be used to provide peak performance optimizations with minor code changes for specialized application scenarios. Typically, only two to three clauses need to be added to the original code.

Check [Python APIs](python_api.md) for details of API functions and [Environment Variables](environment_variables.md) page for environment setting.
   
XPU Engine
-----------------------------------------------------------

Intel® Extension for TensorFlow\* implements a new XPU engine that includes XPU device runtime, graph optimization, and OP/kernel implementation, and brings Intel GPU into the TensorFlow community. This XPU engine also provides deeper performance optimization on Intel CPU hardware. 

You can choose to install Intel GPU backend or CPU backend separately to satisfy different user scenarios.
