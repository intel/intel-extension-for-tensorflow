# Intel® Extension for TensorFlow* for C++

This guide shows how to build an Intel® Extension for TensorFlow* CC library from source and how to work with tensorflow_cc to build bindings for C/C++ languages on Ubuntu.

## Requirements

### Hardware Requirements

Verified Hardware Platforms:
 - Intel® CPU (Xeon, Core)
 - [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html)
 - [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/docs/processors/max-series/overview.html)
 - [Intel® Arc™ Graphics](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) (experimental)

### Common Requirements

#### Install Bazel

To build Intel® Extension for TensorFlow*, install Bazel 5.3.0. Refer to [install Bazel](https://docs.bazel.build/versions/main/install-ubuntu.html).

Here are the recommended commands:

```bash
$ wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-installer-linux-x86_64.sh
$ bash bazel-5.3.0-installer-linux-x86_64.sh --user
```

Check Bazel is installed successfully and is version 5.3.0:

```bash
$ bazel --version
```

#### Download Source Code

```bash
$ git clone https://github.com/intel/intel-extension-for-tensorflow.git intel-extension-for-tensorflow
$ cd intel-extension-for-tensorflow/
```

#### Create a Conda Environment

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Create Virtual Running Environment

```bash
$ conda create -n itex_build python=3.10
$ conda activate itex_build
```

Note, we support Python versions 3.8 through 3.11.

#### Install TensorFlow

Install TensorFlow 2.13.0, and refer to [Install TensorFlow](https://www.tensorflow.org/install) for details.

```bash
$ pip install tensorflow==2.13.0
```

Check TensorFlow was installed successfully and is version 2.13.0:

```bash
$ python -c "import tensorflow as tf;print(tf.__version__)"
```

### Extra Requirements for XPU/GPU Build Only

#### Install Intel GPU Driver
Install the Intel GPU Driver in the building server, which is needed to build with GPU support and AOT ([Ahead-of-time compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html)).

Refer to [Install Intel GPU driver](install_for_xpu.md/#install-gpu-drivers) for details.

Note:

1. Make sure to [install developer runtime packages](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#optional-install-developer-packages) before building Intel® Extension for TensorFlow*.

2. **AOT ([Ahead-of-time compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html))**

    AOT is a compiling option that reduces the initialization time of GPU kernels at startup time by creating the binary code for a specified hardware platform during compiling. AOT will make the installation package larger but improve performance time.

    Without AOT, Intel® Extension for TensorFlow* will be translated to binary code for local hardware platform during startup. That will prolong startup time when using a GPU to several minutes or more.

    For more information, refer to [Use AOT for Integrated Graphics (Intel GPU)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html).

#### Install oneAPI Base Toolkit

We recommend you install the oneAPI base toolkit using `sudo` (or as root user) to the system directory `/opt/intel/oneapi`.

The following commands assume the oneAPI base tookit is installed in `/opt/intel/oneapi`. If you installed it in some other folder, please update the oneAPI path as appropriate.

Refer to [Install oneAPI Base Toolkit Packages](install_for_xpu.md#install-oneapi-base-toolkit-packages)

The oneAPI base toolkit provides compiler and libraries needed by Intel® Extension for TensorFlow*.

Enable oneAPI components:

```bash
$ source /opt/intel/oneapi/compiler/latest/env/vars.sh
$ source /opt/intel/oneapi/mkl/latest/env/vars.sh
```


## Build Intel® Extension for TensorFlow* CC library

### Configure

#### Configure For CPU

Configure the system build by running the `./configure` command at the root of your cloned Intel® Extension for TensorFlow* source tree.

```bash
$ ./configure
```

Choose `n` to build for CPU only. Refer to [Configure Example](how_to_build.md#configure-for-cpu).

#### Configure For GPU

Configure the system build by running the `./configure` command at the root of your cloned Intel® Extension for TensorFlow* source tree. This script prompts you for the location of Intel® Extension for TensorFlow* dependencies and asks for additional build configuration options (path to DPC++ compiler, for example).

```bash
$ ./configure
```

- Choose `Y` for Intel GPU support. Refer to [Configure Example](how_to_build.md#configure-example-for-gpu-or-xpu).

- Specify the Location of Compiler (DPC++).

  Default is `/opt/intel/oneapi/compiler/latest/linux/`, which is the default installed path. Click `Enter` to confirm default location.

  If it's differenct, confirm the compiler (DPC++) installed path and fill the correct path.

- Specify the Ahead of Time (AOT) Compilation Platforms.

  Default is '', which means no AOT.

  Fill one or more device type strings of special hardware platforms, such as `ats-m150`, `acm-g11`.

  Here is the list of GPUs we've verified:

  |GPU|device type|
  |-|-|
  |Intel® Data Center GPU Flex Series 170|ats-m150|
  |Intel® Data Center GPU Flex Series 140|ats-m75|
  |Intel® Data Center GPU Max Series|pvc|
  |Intel® Arc™ A730M|acm-g10|
  |Intel® Arc™ A380|acm-g11|

  To learn how to get the device type, please refer to [Use AOT for Integrated Graphics (Intel GPU)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html) or create an [issue](https://github.com/intel/intel-extension-for-tensorflow/issues) to ask support.

- Choose to Build with oneMKL Support.

  We recommend choosing `y`.

  Default is `/opt/intel/oneapi/mkl/latest`, which is the default installed path. Click `Enter` to confirm default location.

  If it's wrong, please confirm the oneMKL installed path and fill the correct path.


### Build Source Code

For GPU support

```bash
$ bazel build -c opt --config=gpu //itex:libitex_gpu_cc.so
```

CC library location: `<Path to intel-extension-for-tensorflow>/bazel-bin/itex/libitex_gpu_cc.so`

NOTE: `libitex_gpu_cc.so` is depended on `libitex_gpu_xetla.so`, so `libitex_gpu_xetla.so` shoule be copied to the same diretcory of `libitex_gpu_cc.so`
```bash
$ cd <Path to intel-extension-for-tensorflow>
$ cp bazel-out/k8-opt-ST-*/bin/itex/core/kernels/gpu/libitex_gpu_xetla.so bazel-bin/itex/
```

For CPU support

```bash
$ bazel build -c opt --config=cpu //itex:libitex_cpu_cc.so
```

CC library location: `<Path to intel-extension-for-tensorflow>/bazel-bin/itex/libitex_cpu_cc.so`

NOTE: `libitex_cpu_cc.so` is depended on `libiomp5.so`, so `libiomp5.so` shoule be copied to the same diretcory of `libitex_cpu_cc.so`
```bash
$ cd <Path to intel-extension-for-tensorflow>
$ cp bazel-out/k8-opt-ST-*/bin/external/llvm_openmp/libiomp5.so bazel-bin/itex/
```

## Prepare Tensorflow* CC library and header files

### Option 1: Extract from Tensorflow* python package (**Recommended**)

a. Download Tensorflow* 2.13.0 python package

```bash
$ wget https://files.pythonhosted.org/packages/ed/30/310fee0477ce46f722c561dd7e21eebca0d1d29bdb3cf4a2335b845fbba4/tensorflow-2.13.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

b. Unzip Tensorflow* python package

```bash
$ unzip tensorflow-2.13.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -d tensorflow_2.13.0
```

c. Create symbolic link

```bash
$ cd ./tensorflow_2.13.0/tensorflow/
$ ln -s libtensorflow_cc.so.2 libtensorflow_cc.so
$ ln -s libtensorflow_framework.so.2 libtensorflow_framework.so
```
libtensorflow_cc.so location: `<Path to tensorflow_2.13.0>/tensorflow/libtensorflow_cc.so`

libtensorflow_framework.so location: `<Path to tensorflow_2.13.0>/tensorflow/libtensorflow_framework.so`

Tensorflow header file location: `<Path to tensorflow_2.13.0>/tensorflow/include`

### Option 2: Build from TensorFlow* source code

a. Prepare TensorFlow* source code

```bash
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout origin/r2.13 -b r2.13
```

b. Build libtensorflow_cc.so

```bash
$ ./configure
$ bazel build --jobs 96 --config=opt //tensorflow:libtensorflow_cc.so
$ ls ./bazel-bin/tensorflow/libtensorflow_cc.so
```

libtensorflow_cc.so location: `<Path to tensorflow>/bazel-bin/tensorflow/libtensorflow_cc.so`

c. Create symbolic link for libtensorflow_framework.so

```bash
$ cd ./bazel-bin/tensorflow/
$ ln -s libtensorflow_framework.so.2 libtensorflow_framework.so
```

libtensorflow_framework.so location: `<Path to tensorflow>/bazel-bin/tensorflow/libtensorflow_framework.so`

c. Build Tensorflow header files

```bash
$ bazel build --config=opt tensorflow:install_headers
$ ls ./bazel-bin/tensorflow/include
```

Tensorflow header file location: `<Path to tensorflow>/bazel-bin/tensorflow/include`

## Integrate the CC library

### Linker

Configure the linker environmental variables with Intel® Extension for TensorFlow* CC library (**libitex_gpu_cc.so** or **libitex_cpu_cc.so**) path:

```bash
$ export LIBRARY_PATH=$LIBRARY_PATH:<Path to intel-extension-for-tensorflow>/bazel-bin/itex/
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<Path to intel-extension-for-tensorflow>/bazel-bin/itex/
```

### Load

TensorFlow* has C API: `TF_LoadPluggableDeviceLibrary` to support the pluggable device library.
To support Intel® Extension for TensorFlow* cc library, we need to modify the original C++ code:

a. Add the header file: `"tensorflow/c/c_api_experimental.h"`.

```C++
#include "tensorflow/c/c_api_experimental.h"
```

b. Load libitex_gpu_cc.so or libitex_cpu_cc.so by `TF_LoadPluggableDeviceLibrary`.

```C++
TF_Status* status = TF_NewStatus();
TF_LoadPluggableDeviceLibrary(<lib_path>, status);
```

#### Example

The original simple example for using [TensorFlow* C++ API](https://www.tensorflow.org/api_docs/cc).
```c++
// example.cc
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;

  Scope root = Scope::NewRootScope();
  auto X = Variable(root, {5, 2}, DataType::DT_FLOAT);
  auto assign_x = Assign(root, X, RandomNormal(root, {5, 2}, DataType::DT_FLOAT));
  auto Y = Variable(root, {2, 3}, DataType::DT_FLOAT);
  auto assign_y = Assign(root, Y, RandomNormal(root, {2, 3}, DataType::DT_FLOAT));
  auto Z = Const(root, 2.f, {5, 3});
  auto V = MatMul(root, assign_x, assign_y);  
  auto VZ = Add(root, V, Z);

  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch VZ
  TF_CHECK_OK(session.Run({VZ}, &outputs));
  LOG(INFO) << "Output:\n" << outputs[0].matrix<float>();
  return 0;
}
```

The updated example with Intel® Extension for TensorFlow* enabled

```diff
// example.cc
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
+ #include "tensorflow/c/c_api_experimental.h"

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;

+  TF_Status* status = TF_NewStatus();
+  string xpu_lib_path = "libitex_gpu_cc.so";
+  TF_LoadPluggableDeviceLibrary(xpu_lib_path.c_str(), status);
+  TF_Code code = TF_GetCode(status);
+  if ( code == TF_OK ) {
+      LOG(INFO) << "intel-extension-for-tensorflow load successfully!";
+  } else {
+      string status_msg(TF_Message(status));
+      LOG(WARNING) << "Could not load intel-extension-for-tensorflow, please check! " << status_msg;
+  }

  Scope root = Scope::NewRootScope();
  auto X = Variable(root, {5, 2}, DataType::DT_FLOAT);
  auto assign_x = Assign(root, X, RandomNormal(root, {5, 2}, DataType::DT_FLOAT));
  auto Y = Variable(root, {2, 3}, DataType::DT_FLOAT);
  auto assign_y = Assign(root, Y, RandomNormal(root, {2, 3}, DataType::DT_FLOAT));
  auto Z = Const(root, 2.f, {5, 3});
  auto V = MatMul(root, assign_x, assign_y);  
  auto VZ = Add(root, V, Z);

  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch VZ
  TF_CHECK_OK(session.Run({VZ}, &outputs));
  LOG(INFO) << "Output:\n" << outputs[0].matrix<float>();
  return 0;
}
```

### Build and run

Place a `Makefile` file in the same directory of `example.cc` with the following contents:

- Replace `<TF_INCLUDE_PATH>` with local **Tensorflow\* header file path**. e.g.  `<Path to tensorflow_2.13.0>/tensorflow/include`
- Replace `<TFCC_PATH>` with local **Tensorflow\* CC library path**. e.g. `<Path to tensorflow_2.13.0>/tensorflow/`

```Makefile
// Makefile
target = example_test
cc = g++
TF_INCLUDE_PATH = <TF_INCLUDE_PATH>
TFCC_PATH = <TFCC_PATH>
include = -I $(TF_INCLUDE_PATH)
lib = -L $(TFCC_PATH) -ltensorflow_framework -ltensorflow_cc
flag = -Wl,-rpath=$(TFCC_PATH) -std=c++17
source = ./example.cc
$(target): $(source)
	$(cc) $(source) -o $(target) $(include) $(lib) $(flag)
clean:
	rm $(target)
run:
	./$(target)
```

Go to the directory of example.cc and Makefile, then build and run example.

```bash
$ make
$ ./example_test
```

**NOTE:** For GPU support, please set up oneapi environment variables before running the example.

```bash
$ source /opt/intel/oneapi/compiler/latest/env/vars.sh
$ source /opt/intel/oneapi/mkl/latest/env/vars.sh
```