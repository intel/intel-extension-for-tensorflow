Features
===========================================================

Operator Optimization
-----------------------------------------------------------

Intel® Extension for TensorFlow\* optimizes operators in CPU and implements all GPU operators with Intel® oneAPI DPC++ Compiler. Users can get these operator optimization benefits by default without any additional setting. 

Besides, several customized operators for performance boost with `itex.ops` namespace are developed to extend TensorFlow public APIs implementation for better performance. Please refer to `Customized OPs <itex_ops.html>`_ for details.

.. toctree::
   :hidden:
   :maxdepth: 1

   itex_ops.md

Graph Optimization
-----------------------------------------------------------

Intel® Extension for TensorFlow\* provides graph optimization to fuse specified op pattern to new single op for better performance, such as Conv2D+ReLU, Linear+ReLU, etc.  The benefit of the fusions are delivered to users in a transparant fashion. 

Users can get the graph optimization benefits by default without any additional setting. Please refer to `Graph Optimization <itex_fusion.html>`_ for details.

.. toctree::
   :hidden:
   :maxdepth: 1

   itex_fusion.md

Advanced Auto Mixed Precision (AMP)
-----------------------------------------------------------

Low precision data type bfloat16 and float16 are natively supported from the 3rd Generation Xeon® Scalable Processors (aka Cooper Lake) with AVX512 instruction set and Intel® Data Center GPU with further boosted performance and with less memory consumption. The lower-precision data types support of Advanced Auto Mixed Precision (AMP) are fully enabled in Intel® Extension for TensorFlow\*.

Please refer to  `Advanced Auto Mixed Precision <advanced_auto_mixed_precision.html>`_ for details. 

.. toctree::
   :hidden:
   :maxdepth: 1

   advanced_auto_mixed_precision.md

Ease-of-use Python API
-----------------------------------------------------------

Generally, the default configuration of Intel® Extension for TensorFlow\* can get the good performance without any code changes. At the same time, Intel® Extension for TensorFlow\* also provides simple frontend Python APIs and utilities for advanced users to get more performance optimizations with minor code changes for different kinds of application scenarios. Typically, only two to three clauses are required to be added to the original code.

Please check `Python APIs <python_api.html>`_ page for details of API functions and `Environment Variables  <environment_variables.html>`_ page for environment setting.

.. toctree::
   :hidden:
   :maxdepth: 1
   
   python_api.md
   environment_variables.md
   
GPU Profiler
-----------------------------------------------------------

Intel® Extension for TensorFlow\* provides support for TensorFlow* profiler with almost same with TensorFlow Profiler(https://www.tensorflow.org/guide/profiler), one more thing to enable the profiler is exposing three environment variables (`export ZE_ENABLE_TRACING_LAYER=1`, `export UseCyclesPerSecondTimer=1`, `export ENABLE_TF_PROFILER=1`). 

Please refer to `GPU Profiler <how_to_enable_profiler.html>`_ for details.

.. toctree::
   :hidden:
   :maxdepth: 1

   how_to_enable_profiler.md
   
CPU Launcher [Experimental]
-----------------------------------------------------------

There are several factors that influence performance. Setting configuration options properly contributes to a performance boost. However, there is no unified configuration that is optimal to all topologies. Users need to try different combinations.

Intel® Extension for TensorFlow\* provides a CPU launcher to automate these configuration settings to free users from the complicated work. This guide helps you to learn the *launch* script common usage and provides examples that cover many optimized configuration cases as well.

Please refer to `CPU Launcher <launch.html>`_ for details.

.. toctree::
   :hidden:
   :maxdepth: 1

   launch.md
   
INT8 Quantization
-----------------------------------------------------------
Intel® Extension for TensorFlow\* co-works with Intel® Neural Compressor(https://github.com/intel/neural-compressor) to provide compatible TensorFlow INT8 quantization solution support with same user experience. 

Please refer to `INT8 Quantization <INT8_quantization.html>`_ for details.

.. toctree::
   :hidden:
   :maxdepth: 1

   INT8_quantization.md

XPUAutoShard on GPU [Experimental]
-----------------------------------------------------------
Intel® Extension for TensorFlow\* provides XPUAutoShard feature to automatically shard the input data and the TensorFlow graph, placing these data/graph shards on GPU devices to maximize the hardware usage. 

Please refer to `XPUAutoShard <XPUAutoShard.html>`_ for details.

.. toctree::
   :hidden:
   :maxdepth: 1

   XPUAutoShard.md