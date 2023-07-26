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


#!/usr/bin/env python
# coding: utf-8

import json
import time
import os
import sys
import numpy as np

import compare_result
import benchmark

import intel_extension_for_tensorflow as itex
print("intel_extension_for_tensorflow {}".format(itex.__version__))

def set_itex_fp32(device):
    print("Set itex for FP32 with backend {}".format(device))

def set_itex_amp(amp_target):
    # set configure for auto mixed precision.
    auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()
    if amp_target=="BF16":
        auto_mixed_precision_options.data_type = itex.BFLOAT16
    else:
        auto_mixed_precision_options.data_type = itex.FLOAT16

    graph_options = itex.GraphOptions(auto_mixed_precision_options=auto_mixed_precision_options)
    # enable auto mixed precision.
    graph_options.auto_mixed_precision = itex.ON

    config = itex.ConfigProto(graph_options=graph_options)
    itex.set_config(config)

    print("Set itex for AMP (auto_mixed_precision, {}_FP32) with backend {}".format(amp_target, device))


def main(device, amp):
   
    set_itex_fp32(device)
    bench_res = benchmark.benchmark()
    
    bench_res_file = "fp32_bench_res.json"
    benchmark.save_json_data(bench_res, bench_res_file)
    
    
    set_itex_amp(amp)
    bench_res = benchmark.benchmark()
    bench_res_file = "amp_bench_res.json"
    benchmark.save_json_data(bench_res, bench_res_file)

    print("Benchmark is done!")

    compare_result.compare(device=device, target=amp)

    print("Finished")


def help(me):
    print("{} cpu|gpu bf16|fp16".format(me))

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Miss parameters!")
        help(sys.argv[0])
        sys.exit(1)
    me = sys.argv[0]
    device = sys.argv[1].lower()
    amp = sys.argv[2].lower()

    if device not in ['cpu', 'gpu']:
        print("Parameter value is wrong!")
        help(me)
        sys.exit(1)

    if amp not in ['bf16', 'fp16']:
        print("Parameter value is wrong!")
        help(me)
        sys.exit(1)
    amp = amp.upper()
    device = device.upper()
    main(device, amp)
