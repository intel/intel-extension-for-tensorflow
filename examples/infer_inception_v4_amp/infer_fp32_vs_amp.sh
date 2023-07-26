# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#!/bin/bash


if [[ $# -ne 2 ]]; then
    echo "Miss parameter"
    echo "$0 cpu|gpu bf16|fp16"
    exit 1
fi

device_type=$1
if [ ${device_type} != "cpu" ] && [ ${device_type} != "gpu" ]; then
    echo "Wrong parameter"
    echo "$0 cpu|gpu"
    exit 2
fi

target_type=$2
if [ ${target_type} != "bf16" ] && [ ${target_type} != "fp16" ]; then
    echo "Wrong parameter"
    echo "$0 bf16|fp16"
    exit 2
fi

#use PyPi to setup a virtual environment
ENV_NAME=env_itex_${device_type}
if [ ! -d $ENV_NAME ]; then
    echo "Create env $ENV_NAME ..."
    bash set_env_${device_type}.sh
else
    echo "Already created env $ENV_NAME, skip creating env"
fi

source $ENV_NAME/bin/activate

echo "Infer with FP32 on ${device_type}"
python benchmark.py ${device_type}
mv bench_res.json fp32_bench_res.json

if [ ${target_type} == "bf16" ]; then
    target_key="BFLOAT16"
else
    target_key="FLOAT16"
fi

echo "Set Environment Variables for ${target_key}"
export ITEX_AUTO_MIXED_PRECISION=1

export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE=${target_key}

echo "InceptionV4 Inference for ${target_key} on ${device_type}"
python benchmark.py ${device_type}

mv bench_res.json amp_bench_res.json

echo "Compare Result"
python compare_result.py ${target_type}
