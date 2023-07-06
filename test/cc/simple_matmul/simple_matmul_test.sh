#!/bin/bash
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

function test_case_build {
    tf_dir=$1
    test_target=$2

    # Prepare depended Tensorflow* libraries
    if [ ! -d $(readlink -f $tf_dir) ]; then
        echo "Could not find Tensorflow* path: $tf_dir, please check!"
        exit 1
    fi
    tf_cc_lib_name=$(ls $tf_dir | grep 'libtensorflow_cc.so.')
    if [ ! -z "$tf_cc_lib_name" ] && [ -f "${tf_dir}/${tf_cc_lib_name}" ]; then
        if [ ! -L "${tf_dir}/libtensorflow_cc.so" ] || [ $(readlink -f "${tf_dir}/libtensorflow_cc.so") != "${tf_dir}/${tf_cc_lib_name}" ]; then
            ln -sf "${tf_dir}/${tf_cc_lib_name}" "${tf_dir}/libtensorflow_cc.so"
        fi
    else
        echo "Could not find libtensorflow_cc.so from $tf_dir, please check!"
        exit 1
    fi
    tf_fw_lib_name=$(ls $tf_dir | grep 'libtensorflow_framework.so.')
    if [ ! -z "$tf_fw_lib_name" ] && [ -f "${tf_dir}/${tf_fw_lib_name}" ]; then
        if [ ! -L "${tf_dir}/libtensorflow_framework.so" ] || [ $(readlink -f "${tf_dir}/libtensorflow_framework.so") != "${tf_dir}/${tf_fw_lib_name}" ]; then
            ln -sf "${tf_dir}/${tf_fw_lib_name}" "${tf_dir}/libtensorflow_framework.so"
        fi
    else
        echo "Could not find libtensorflow_framework.so from $tf_dir, please check!"
        exit 1
    fi

    # Prepare Makefile
    if [ "$test_target" == "CPU" ]; then
        build_target="ITEX_CPU_CC"
        make_file="Makefile.cpu"
    else
        build_target="ITEX_GPU_CC"
        make_file="Makefile.gpu"
    fi
    sed -e "s#<TF_PATH>#$tf_dir#g" -e "s#<BUILD_TARGET>#$build_target#g" Makefile.tpl > $make_file

    # Build
    [ -f simple_matmul ] && rm -f simple_matmul
    make -f $make_file >& build.log
    if [ $? -eq 0 ]; then
        echo "Simple matmul C++ example is built successfully!"
    else
        echo "Simple matmul C++ example is built failed! please check build.log!"
        exit 1
    fi
}

function run_test_case {
    itex_cc_dir=$1
    test_target=$2

    if [ "$test_target" == "CPU" ] && [ -f "${itex_cc_dir}/libitex_cpu_cc.so" ]; then
        echo "Starting to run Simple matmul C++ example with intel_extension_for_tensorflow CPU."
    elif [ "$test_target" == "GPU" ] && [ -f "${itex_cc_dir}/libitex_gpu_cc.so" ]; then
        echo "Starting to run Simple matmul C++ example with intel_extension_for_tensorflow GPU."
    else
        echo "Could not find the intel_extension_for_tensorflow CC library (Type: $test_target) from $itex_cc_dir, please check!"
        exit 1
    fi

    label_image.log.${test_target}
    export LD_LIBRARY_PATH=$itex_cc_dir:$LD_LIBRARY_PATH
    export ITEX_VERBOSE=3
    ./simple_matmul >& simple_matmul.log.${test_target}
}

function check_result {
    if [ $(grep " Output:" simple_matmul.log.${test_target} >& /dev/null; echo $?) -eq 0 ]; then
        echo "Simple matmul test [Passed]"
    else
        echo "Simple matmul test [Failed]"
        exit 1
    fi
}

function help {
    cat <<EOF
Usage: $0 -i <ITEX CC DIR> -f <TF CC DIR> -t <CPU|GPU>
  <ITEX CC DIR>: The directory path to the libitex_gpu_cc.so or libitex_cpu_cc.so
  <TF CC DIR>:   The directory path to the libtensorflow_cc.so.* and libtensorflow_framework.so.*
  <CPU|GPU>:     The flag to specifiy which type of test case to build, CPU or GPU
EOF
    exit 1
}


function main {
    tf_cc_dir=$1
    itex_cc_dir=$2
    target=$3

    # Build
    test_case_build $tf_cc_dir $target

    # Test
    run_test_case $itex_cc_dir $target

    # Check
    check_result

    exit 0
}

while getopts "ht:i:f:" arg; do
    case $arg in
        t)
          type=$OPTARG
          ;;
        i)
          itex_dir=$OPTARG
          ;;
        f)
          tf_dir=$OPTARG
          ;;
        h|*)
          help
          exit 1
          ;;
    esac
done

if [ -z "$tf_dir" ] || [ -z "$itex_dir" ] || [ -z "$type" ]; then
    help
    exit 1
fi
if [ "$type" != "CPU" ] && [ "$type" != "GPU" ]; then
    echo "illegal value for \"-t\" -- $type, only support \"CPU\" or \"GPU\""
    exit 1
fi

main $tf_dir $itex_dir $type
