#!/usr/bin/env bash
#
# Copyright (c) 2021-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e
IMAGE_TYPE=$1
if [ $IMAGE_TYPE == "gpu" ]
then
        IMAGE_NAME=intel-extension-for-tensorflow:gpu
        docker build --build-arg UBUNTU_VERSION=20.04 \
                                --build-arg PYTHON=python3.9 \
                                --build-arg ICD_VER=22.28.23726.1+i419~u20.04 \
                                --build-arg LEVEL_ZERO_GPU_VER=1.3.23726.1+i419~u20.04 \
                                --build-arg LEVEL_ZERO_VER=1.8.1+i755~u20.04 \
                                --build-arg TF_VER=2.10 \
                                --build-arg DPCPP_VER=2022.2.0-8734 \
                                --build-arg MKL_VER=2022.2.0-8748 \
                                --build-arg TF_PLUGIN_WHEEL=intel_extension_for_tensorflow*.whl \
                                -t $IMAGE_NAME \
				-f itex-gpu.Dockerfile .
elif  [ $IMAGE_TYPE == "cpu-centos" ]
then
        IMAGE_NAME=intel-extension-for-tensorflow:cpu-centos
        docker build --build-arg CENTOS_VER=8 \
                                --build-arg PY_VER=39 \
                                --build-arg PYTHON=python3.9 \
                                --build-arg TF_VER=2.10 \
                                --build-arg TF_PLUGIN_WHEEL=intel_extension_for_tensorflow*.whl \
                                -t $IMAGE_NAME \
                                -f itex-cpu-centos.Dockerfile .
else
        IMAGE_NAME=intel-extension-for-tensorflow:cpu-ubuntu
        docker build --build-arg UBUNTU_VERSION=20.04 \
                                --build-arg PYTHON=python3.9 \
                                --build-arg TF_VER=2.10 \
                                --build-arg TF_PLUGIN_WHEEL=intel_extension_for_tensorflow*.whl \
                                -t $IMAGE_NAME \
                                -f itex-cpu-ubuntu.Dockerfile .
fi

