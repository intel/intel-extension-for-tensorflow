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
      <td colspan="3" align="center"><a href="guide/infrastructure.md">Infrastructure</a></td>
      <td colspan="3" align="center"><a href="../examples/quick_example.md">Quick example</a></td>
      <td colspan="3" align="center"><a href="../examples">Examples</a></td>
      <td colspan="3" align="center"><a href="community/releases.md">Releases</a></td>
    </tr>
    <tr>
      <td colspan="6" align="center"><a href="guide/FAQ.md">Frequently asked questions</a></td>
      <td colspan="6" align="center"><a href="community/contributing.md">Contributing guidelines</a></td>
    </tr>
  </tbody>
  <thead>
  <tr>
    <th colspan="12">Installation guide</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3" align="center"><a href="install/experimental/install_for_cpu.md">Install for CPU</a></td>
      <td colspan="3" align="center"><a href="install/install_for_gpu.md">Install for GPU</a></td>
      <td colspan="3" align="center"><a href="install/how_to_build.md">Install by source build</a></td>
	  <td colspan="3" align="center"><a href="install/experimental/install_for_gpu_conda.md">Install Conda for GPU distributed</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="12">Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="3" align="center"><a href="guide/environment_variables.md">Environment variables</a></td>
		<td colspan="3" align="center"><a href="guide/python_api.md">Python API</a></td>
        <td colspan="3" align="center"><a href="guide/advanced_auto_mixed_precision.md">Advanced auto mixed precision</a></td>
		<td colspan="3" align="center"><a href="guide/itex_ops.md">Operator optimization</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><a href="guide/itex_fusion.md">Graph optimization</a></td>
        <td colspan="3" align="center"><a href="guide/how_to_enable_profiler.md">GPU profiler</a></td>
		<td colspan="3" align="center"><a href="guide/launch.md">CPU launcher</a></td>
        <td colspan="3" align="center"><a href="guide/INT8_quantization.md">INT8 quantization</a></td>
    </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="12">Advanced topics</th>
      </tr>
  </thead>
  <tbody>
      <tr>
        <td colspan="4" align="center"><a href="guide/practice_guide.md#cpu-practice-guide">CPU practice guide</a></td>
        <td colspan="4" align="center"><a href="guide/practice_guide.md#gpu-practice-guide">GPU practice guide</a></td>
        <td colspan="4" align="center"><a href="install/experimental/install_for_cpp.md">C++ API support</a></td>
      </tr>
  </tbody>
    <thead>
      <tr>
        <th colspan="12">Developer Guide</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="4" align="center"><a href="design/extension_design.md">Extension design</a></td>
		  <td colspan="4" align="center"><a href="design/directory_structure.md">Directory structure</a></td>
		  <td colspan="4" align="center"><a href="design/optimization/README.md">Optimizations design</a></td>
      </tr>
  </tbody>
</table>


## Hightlights

* Environment variables & Python API

  Generally, the default configuration of Intel® Extension for TensorFlow\* will provide good performance without any code changes. 
  Intel® Extension for TensorFlow\* also provides simple frontend Python APIs and utilities, for advanced users to get more optimized performance with only minor code changes, for different kinds of application scenarios. Typically, only two or three clauses are required to be added to the original code.

* Advanced auto mixed precision (AMP)

  Low precision data types `bfloat16` and` float16` are natively supported from the `3rd Generation Xeon® Scalable Processors` [Cooper Lake](https://ark.intel.com/content/www/us/en/ark/products/series/204098/3rd-generation-intel-xeon-scalable-processors.html)  with `AVX512` instruction set and Intel® Data Center GPU with further boosts to performance and with less memory consumption. The lower-precision data types supported by Advanced Auto Mixed Precision (AMP) are fully enabled in Intel® Extension for TensorFlow*.

* Graph optimization

  Intel® Extension for TensorFlow\* provides graph optimization to fuse specific operator patterns to a new single operator for better performance, such as `Conv2D+ReLU` or `Linear+ReLU`.  The benefits of the fusions are delivered to users in a transparent fashion.

* Operator optimization

  Intel® Extension for TensorFlow\* also optimizes operators and implements several customized operators for performance boost. The `itex.ops` namespace is used to extend TensorFlow public APIs implementation for better performance.

* GPU profiler

  Intel® Extension for TensorFlow\* provides support for TensorFlow [Profiler](https://www.tensorflow.org/guide/profiler). To enable the profiler, define three environment variables ( `export ZE_ENABLE_TRACING_LAYER=1`, `export UseCyclesPerSecondTimer=1`, `export ENABLE_TF_PROFILER=1`)

* INT8 quantization

  Intel® Extension for TensorFlow* co-works with [Intel® Neural Compressor](https://github.com/intel/neural-compressor) to provide compatible TensorFlow INT8 quantization solution support with equivalent user experience.
