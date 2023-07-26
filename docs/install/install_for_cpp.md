# Intel® Extension for TensorFlow* for C++

This guide shows how to build an Intel® Extension for TensorFlow* CC library from source and how to work with tensorflow_cc to build bindings for C/C++ languages on Ubuntu 20.04 (64-bit).

## Prepare

Refer to [Build from Source Code -> Prepare](../how_to_build.md#prepare)

## Configure the build

Refer to [Build from Source Code -> Configure the build](../how_to_build.md#configure-the-build)

## Build the CC library

### GPU support

```bash
$ bazel build -c opt --config=gpu //itex:libitex_gpu_cc.so
```

CC library location: `<Path to intel-extension-for-tensorflow>/bazel-bin/itex/libitex_gpu_cc.so`

### CPU support

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
$ source /opt/intel/oneapi/tbb/latest/env/vars.sh
```
