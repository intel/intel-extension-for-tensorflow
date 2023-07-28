# Install TensorFlow Serving with Intel® Extension for TensorFlow*

[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) is an open-source system designed by Google that acts as a bridge between trained machine learning models and the applications that need to use them, streamlining the process of deploying and serving models in a production environment while maintaining efficiency and scalability.

## Install Model Server

### Install using Docker
A good way to get started using TensorFlow Serving with Intel® Extension for TensorFlow* is with [Docker](https://www.docker.com/) containers.
- Install Docker on Ubuntu 22.04
  ```
  sudo apt install docker
  sudo apt install docker.io
  ```
- Pull docker image
  ```
  # For CPU
  docker pull intel/intel-extension-for-tensorflow:serving-cpu

  # For GPU
  docker pull intel/intel-extension-for-tensorflow:serving-gpu 
  ```

### Build from source
> **Tips**:   
> - We recommend you put the source code of Intel® Extension for TensorFlow*, TensorFlow, and TensorFlow Serving in the same folder.
> - Replace related paths with those on your machine.

#### 1. Build Intel® Extension for TensorFlow* C++ library
Refer to [Intel® Extension for TensorFlow* for C++](https://intel.github.io/intel-extension-for-tensorflow/latest/docs/install/experimental/install_for_cpp.html) to build Intel® Extension for TensorFlow* C++ library
> **Note:** When following this installation guide, you only need to build the Intel® Extension for TensorFlow* C++ library. You can ignore the other steps.  

The generated `libitex_cpu_cc.so` or `libitex_gpu_cc.so` binary are found in the `intel_extension_for_tensorflow/bazel-bin/itex/` directory.

#### 2. Build TensorFlow Serving
- Patch TensorFlow
  - Get TensorFlow with commit id specified by TensorFlow Serving: https://github.com/tensorflow/serving/blob/master/WORKSPACE#L28
    ```
    # Exit intel-extension-for-tensorflow source code folder
    cd ..

    # clone TensorFlow
    git clone https://github.com/tensorflow/tensorflow

    # checkout specific commit id
    cd tensorflow   
    git checkout xxxxx
    ```
  - Add `alwayslink=1` for `kernels_experimental` library in local `tensorflow/tensorflow/c/BUILD` file:
    ```
    tf_cuda_library(
        name = "kernels_experimental",
        srcs = ["kernels_experimental.cc"],
        hdrs = ["kernels_experimental.h"],
        copts = tf_copts(),
        visibility = ["//visibility:public"],
        deps = [
            ...
        ] + if_not_mobile([
            ...
        ]),
        alwayslink=1, # add this line
    )
    ```
- Patch TensorFlow Serving
  - Get TensorFlow Serving source code
    ```
    # Exit tensorflow source code folder
    cd ..

    git clone https://github.com/tensorflow/serving
    ```
  - Patch TensorFlow Serving
    ```
    cd serving
    patch -p1 -i ../intel-extension-for-tensorflow/third_party/tf_serving/serving_plugin.patch
    ```
  - Update `serving/WORKSPACE` to use local TensorFlow  
    Replace L24-L29 with below code to use local TensorFlow: https://github.com/tensorflow/serving/blob/master/WORKSPACE#L24   
    ```
    local_repository(
        name= "org_tensorflow",
        path = "path to local tensorflow source code",
    )
    ```

- Build TensorFlow Serving
  ```
  bazel build --copt="-Wno-error=stringop-truncation" --config=release //tensorflow_serving/model_servers:tensorflow_model_server
  ```
  The generated `tensorflow_model_server` will be found in the `serving/bazel-bin/tensorflow_serving/model_servers/` directory. 

### Build Docker image from Dockerfile
Refer to [Intel® Extension for TensorFlow* Serving Docker Container Guide](../../docker/tensorflow-serving/README.md) to build docker image from dockerfile.

## Run sample
- Train and export TensorFlow model
  ```
  cd serving
  rm -rf /tmp/mnist
  python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist
  ```
  Now let's take a look at the export directory. You should find a directory named `1` that contains `saved_models.pb` file and `variables` folder.
  ```
  ls /tmp/mnist
  1

  ls /tmp/mnist/1
  saved_model.pb variables
  ```
- Load exported model with TensorFlow ModelServer plugged with Intel® Extension for TensorFlow* 
  - Use Docker from Docker Hub
    ```
    # For CPU
    docker run \
      -it \
      --rm \
      -p 8500:8500 \
      -e MODEL_NAME=mnist \
      -v /tmp/mnist:/models/mnist \
      intel/intel-extension-for-tensorflow:serving-cpu

    # For GPU
    docker run \
      -it \
      --rm \
      -p 8500:8500 \
      -e MODEL_NAME=mnist \
      -v /tmp/mnist:/models/mnist \
      --device /dev/dri/ \
      -v /dev/dri/by-path/:/dev/dri/by-path/ \
      intel/intel-extension-for-tensorflow:serving-gpu
    ```
    You will see:
    ```
    plugin library "/itex/bazel-bin/itex/libitex_cpu_cc.so" load successfully!

    plugin library "/itex/bazel-bin/itex/libitex_gpu_cc.so" load successfully!
    ```

  - Use tensorflow_model_server built from source
    ```
    # cd tensorflow_model_server binary folder

     # For CPU
    ./tensorflow_model_server \
      --port=8500 \
      --rest_api_port=8501 \
      --model_name=mnist \
      --model_base_path=/tmp/mnist \
      --tensorflow_plugins=path_to_libitex_cpu_cc.so

    # For GPU
    # source oneapi environment
    source oneapi_install_path/compiler/latest/env/vars.sh
    source oneapi_install_path/mkl/latest/env/vars.sh

    ./tensorflow_model_server \
      --port=8500 \
      --rest_api_port=8501 \
      --model_name=mnist \
      --model_base_path=/tmp/mnist \
      --tensorflow_plugins=path_to_libitex_gpu_cc.so
    ```
    You will see:  
    ```
    plugin library "path_to_libitex_cpu_cc.so/libitex_cpu_cc.so" load successfully!
 
    plugin library "path_to_libitex_gpu_cc.so/libitex_gpu_cc.so" load successfully!
    ```

  - Use Docker built from dockerfile
    ```
    cd intel-extension-for-tensorflow source code folder

    cd docker/tensorflow-serving
    
    export MODEL_NAME=mnist
    export MODEL_DIR=/tmp/mnist
    
    ./run.sh [cpu/gpu]
    ```
    You will see:  
    ```
    plugin library "/itex/itex-bazel-bin/bin/itex/libitex_cpu_cc.so" load successfully!
    
    plugin library "/itex/itex-bazel-bin/bin/itex/libitex_gpu_cc.so" load successfully!
    ```

- Test the server
  ```
  pip install tensorflow-serving-api

  cd serving
  python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=127.0.0.1:8500
  ```
  You will see:
  ```
  ...

  Inference error rate: xx.xx%
  ```

Refer to [TensorFlow Serving Guides](https://www.tensorflow.org/tfx/serving/serving_basic) to learn more about how to use TensorFlow Serving.
