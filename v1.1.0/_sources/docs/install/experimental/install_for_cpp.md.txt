# Intel® Extension for TensorFlow* for C++

This guide shows how to build an Intel® Extension for TensorFlow* CC library from source and how to work with tensorflow_cc to build bindings for C/C++ languages on ubuntu 20.04 (64-bit).

## Prepare

Refer to [Build from Source Code -> Prepare](../how_to_build.md#prepare)

## Configure the build

Refer to [Build from Source Code -> Configure the build](../how_to_build.md#configure-the-build)

## Build the CC library

### GPU support

```bash
$ bazel build -c opt --config=gpu //itex:itex_gpu_cc
```

CC library location: `<Path to intel-extension-for-tensorflow>/bazel-bin/itex/libitex_gpu_cc.so`

### CPU only (experimental)

```bash
$ bazel build -c opt --config=cpu //itex:itex_cpu_cc
```

CC library location: `<Path to intel-extension-for-tensorflow>/bazel-bin/itex/libitex_cpu_cc.so`

## Build libtensorfow_cc.so

1. Prepare TensorFlow* source code

```bash
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout origin/r2.10 -b r2.10
```

Apply patch as below to support **TF_LoadPluggableDeviceLibrary**, which is defined in "//tensorflow/c:c_api_experimental"
[A feature request](https://github.com/tensorflow/tensorflow/issues/58533) has been reported to TensorFlow*. Once it is solved, this patch is no longer needed.

```diff
diff --git a/tensorflow/BUILD b/tensorflow/BUILD
index 19ee8000206..77d8c714729 100644
--- a/tensorflow/BUILD
+++ b/tensorflow/BUILD
@@ -1187,6 +1187,7 @@ tf_cc_shared_library(
     visibility = ["//visibility:public"],
     win_def_file = ":tensorflow_filtered_def_file",
     deps = [
+        "//tensorflow/c:c_api_experimental",
         "//tensorflow/c:c_api",
         "//tensorflow/c:env",
         "//tensorflow/c:kernels",
```

2. Build libtensorflow_cc.so

```bash
$ ./configure
$ bazel build --jobs 96 --config=opt //tensorflow:libtensorflow_cc.so
$ ls ./bazel-bin/tensorflow/libtensorflow_cc.so
```

## Integrate the CC library

### Linker

If you place the Intel® Extension for TensorFlow* CC library to a non-system directory, such as ~/mydir, then configure the linker environmental variables:

```bash
$ export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib
```

### Load

TensorFlow* has C API: `TF_LoadPluggableDeviceLibrary` to support the pluggable device library.
To support Intel® Extension for TensorFlow* cc library, we need to modify the orginal C++ code:

1. Add the header file: `"tensorflow/c/c_api_experimental.h"`.

```C++
#include "tensorflow/c/c_api_experimental.h"
```

2. Load libitex_gpu_cc.so or libitex_cpu_cc.so by `TF_LoadPluggableDeviceLibrary`.

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

- Replace `<TF_INCLUDE_PATH>` with local tensorflow include path.
- Replace `<TFCC_PATH>` with local tensorflow_cc path, the path of libtensorflow_cc.so.

```Makefile
// Makefile
target = example_test
cc = g++
TF_INCLUDE_PATH = <TF_INCLUDE_PATH>
TFCC_PATH = <TFCC_PATH>
include = -I $(TF_INCLUDE_PATH)
lib = -L $(TFCC_PATH) -ltensorflow_framework -ltensorflow_cc
flag = -Wl,-rpath=$(TFCC_PATH)
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

**NOTE:** For GPU support, please setup oneapi environment variables before run.

```bash
$ source /opt/intel/oneapi/setvars.sh
```
