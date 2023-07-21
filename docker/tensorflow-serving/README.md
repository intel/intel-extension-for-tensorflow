# Intel® Extension for TensorFlow* Serving Docker Container Guide

## Description

This document has instruction for running TensorFlow Serving using Intel® Extension for TensorFlow* in docker container.

## Build the Docker Image

To build the docker container, enter into [docker/tensorflow-serving](./) folder and follow the below steps.

### I. Binaries Preparation

Refer to [Install for Tensorflow Serving](../../docs/guide/tensorflow_serving.md) to build Tensorflow Serving binary, and refer to [Install for CPP](../../docs/install/install_for_cpp.md) to build Intel® Extension for TensorFlow* CC library form source. And then package and copy them into ./models/binaries directory.

```bash
mkdir -p ./models/binaries

# Package and copy Intel® Extension for TensorFlow* CC library
mkdir -p itex-bazel-bin/
cp -r <path_to_itex>/bazel-out/k8-opt-ST-*/bin/ itex-bazel-bin/
tar cvfh itex-bazel-bin.tar itex-bazel-bin/
cp itex-bazel-bin.tar  ./models/binaries/

# Copy Tensorflow Serving binary
cp path_to_tensorflow_serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server ./models/binaries/

```

### II. Build the Container

If you build the container using Intel GPU, make sure you meet below assumptions:

* Host machine installs Intel GPU.
* Host machine installs Linux kernel that is compatible with GPU drivers.
* Host machine has Intel GPU driver.

Refer to [Install for GPU](../docs/install/install_for_xpu.md) for detail.

Run the [build.sh](./build.sh) to build target docker image.
```bash
./build.sh [gpu/cpu]
```

## Running the Container

Run following commands to start docker container. You can use `-v` option to mount your local directory into container. To make GPU available in the container, attach the GPU to the container using `--device /dev/dri` option and run the container:

```
IMAGE_NAME=intel-extension-for-tensorflow:serving-gpu
docker run -v <your-local-dir>:/workspace \
           -v /dev/dri/by-path:/dev/dri/by-path \
           --device /dev/dri \
           --privileged \
           --ipc=host \
           -p 8500:8500 \
           -e MODEL_NAME=<your-model-name> \
           -e MODEL_DIR=<your-model-dir> \
           -e http_proxy=$http_proxy \
           -e https_proxy=$https_proxy \
           -e no_proxy=$no_proxy \
           -it \
           $IMAGE_NAME
```
