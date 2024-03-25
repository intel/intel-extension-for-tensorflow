# Welcome to Intel® Extension for TensorFlow* documentation


## Documentation

<table class="docutils">
  <thead>
  <tr>
    <th colspan="12">Overview</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3" align="center"><a href="guide/infrastructure.html">Infrastructure</a></td>
      <td colspan="3" align="center"><a href="../examples/quick_example.html">Quick example</a></td>
      <td colspan="3" align="center"><a href="../examples">Examples</a></td>
      <td colspan="3" align="center"><a href="community/releases.html">Releases</a></td>
    </tr>
    <tr>
      <td colspan="3" align="center"><a href="guide/performance.html">Performance data</a></td>
      <td colspan="6" align="center"><a href="guide/FAQ.html">Frequently asked questions</a></td>
      <td colspan="3" align="center"><a href="community/contributing.html">Contributing guidelines</a></td>
    </tr>
  </tbody>
  <thead>
  <tr>
    <th colspan="12">Installation guide</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3" align="center"><a href="install/install_for_cpu.html">Install for CPU</a></td>
      <td colspan="3" align="center"><a href="install/install_for_xpu.html">Install for XPU</a></td>
      <td colspan="3" align="center"><a href="install/how_to_build.html">Install by source build</a></td>
	  <td colspan="3" align="center"><a href="install/experimental/install_for_gpu_conda.html">Install Conda for GPU distributed</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="12">Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="2" align="center"><a href="guide/environment_variables.html">Environment variables</a></td>
	    	<td colspan="2" align="center"><a href="guide/python_api.html">Python API</a></td>
        <td colspan="4" align="center"><a href="guide/next_pluggable_device.html">Next Pluggable Device</a></td>
        <td colspan="2" align="center"><a href="guide/threadpool.html">CPU Thread Pool</a></td>
    </tr>
    <tr>
        <td colspan="2" align="center"><a href="guide/itex_fusion.html">Graph optimization</a></td>
        <td colspan="2" align="center"><a href="guide/itex_ops.html">Custom operator</a></td>
        <td colspan="4" align="center"><a href="guide/advanced_auto_mixed_precision.html">Advanced auto mixed precision</a></td>
	      <td colspan="2" align="center"><a href="guide/itex_ops_override.html">Operator override</a></td>
    </tr>
    <tr>    
	      <td colspan="3" align="center"><a href="guide/INT8_quantization.html">INT8 quantization</a></td>
	      <td colspan="2" align="center"><a href="guide/XPUAutoShard.html">XPUAutoShard</a></td>
        <td colspan="2" align="center"><a href="guide/how_to_enable_profiler.html">GPU profiler</a></td>
	      <td colspan="2" align="center"><a href="guide/launch.html">CPU launcher</a></td>
      	<td colspan="2" align="center"><a href="guide/weight_prepack.html">Weight prepack</a></td>
    </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="12">Advanced topics</th>
      </tr>
  </thead>
  <tbody>
      <tr>
        <td colspan="3" align="center"><a href="guide/practice_guide.html#cpu-practice-guide">CPU practice guide</a></td>
        <td colspan="3" align="center"><a href="guide/practice_guide.html#gpu-practice-guide">GPU practice guide</a></td>
        <td colspan="3" align="center"><a href="install/install_for_cpp.html">C++ API support</a></td>
        <td colspan="3" align="center"><a href="guide/OpenXLA_Support_on_GPU.html">OpenXLA Support on GPU</a></td>
      </tr>
  </tbody>
    <thead>
      <tr>
        <th colspan="12">Developer Guide</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="3" align="center"><a href="design/extension_design.html">Extension design</a></td>
	  <td colspan="3" align="center"><a href="design/directory_structure.html">Directory structure</a></td>
	  <td colspan="3" align="center"><a href="design/optimization/README.html">Optimizations design</a></td>
          <td colspan="3" align="center"><a href="design/how_to_write_custom_op.html">Custom Op</a></td>
      </tr>
  </tbody>
</table>


## Highlights

* Environment variables & Python API

  Generally, the default configuration of Intel® Extension for TensorFlow\* provides good performance without any code changes. 
  Intel® Extension for TensorFlow\* also provides simple frontend Python APIs and utilities for advanced users to get more optimized performance with only minor code changes for different kinds of application scenarios. Typically, you only need to add two or three clauses to the original code.

* Next Pluggable Device (NPD)
  
  The Next Pluggable Device (NPD) represents an advanced generation of TensorFlow plugin mechanisms. It not only facilitates a seamless integration of new accelerator plugins for registering devices with TensorFlow without requiring modifications to the TensorFlow codebase, but it also serves as a conduit to OpenXLA via its PJRT plugin. This innovative approach significantly streamlines the process of extending TensorFlow's capabilities with new hardware accelerators, enhancing both efficiency and flexibility.
  
* Advanced auto mixed precision (AMP)

  Low precision data types `bfloat16` and` float16` are natively supported by the `3rd Generation Xeon® Scalable Processors`, codenamed [Cooper Lake](https://ark.intel.com/content/www/us/en/ark/products/series/204098/3rd-generation-intel-xeon-scalable-processors.html),  with `AVX512` instruction set and the Intel® Data Center GPU, which further boosts performance and uses less memory. The lower-precision data types supported by Advanced Auto Mixed Precision (AMP) are fully enabled in Intel® Extension for TensorFlow*.

* Graph optimization

  Intel® Extension for TensorFlow\* provides graph optimization to fuse specific operator patterns to a new single operator for better performance, such as `Conv2D+ReLU` or `Linear+ReLU`.  The benefits of the fusions are delivered to users in a transparent fashion.

* CPU Thread Pool

  Intel® Extension for TensorFlow\* uses OMP thread pool by default since it has better performance and scaling for most cases. For workloads with large inter-op concurrency, you can switch to use Eigen thread pool (default in TensorFlow) by setting the environment variable `ITEX_OMP_THREADPOOL=0`.

* Operator optimization

  Intel® Extension for TensorFlow\* also optimizes operators and implements several customized operators for a performance boost. The `itex.ops` namespace is used to extend TensorFlow public APIs implementation for better performance.

* GPU profiler

  Intel® Extension for TensorFlow\* provides support for TensorFlow [Profiler](https://www.tensorflow.org/guide/profiler). To enable the profiler, define three environment variables ( `export ZE_ENABLE_TRACING_LAYER=1`, `export UseCyclesPerSecondTimer=1`, `export ENABLE_TF_PROFILER=1`)

* INT8 quantization

  Intel® Extension for TensorFlow* co-works with [Intel® Neural Compressor](https://github.com/intel/neural-compressor) to provide compatible TensorFlow INT8 quantization solution support with equivalent user experience.

* XPUAutoShard on GPU [Experimental]

  Intel® Extension for TensorFlow\* provides XPUAutoShard feature to automatically shard the input data and the TensorFlow graph, placing these data/graph shards on GPU devices to maximize the hardware usage.

* OpenXLA Support on GPU [Experimental]

  Intel® Extension for TensorFlow\* adopts a uniform Device API PJRT as the supported device plugin mechanism to implement Intel GPU backend for OpenXLA experimental support.
