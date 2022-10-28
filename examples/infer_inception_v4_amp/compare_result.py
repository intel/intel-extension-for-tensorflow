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

def load_json_data(filename="train.json"):
    with open(filename, "r") as f:
        return json.load(f)

def fix_len(name, length):
    if len(name)<length:
        name+=(" "*(length-len(name)))
    return name
    
def format_print(name, values):
    [a, b] =values
    a=str(a)
    b=str(b)
    name = fix_len(name, 32)
    a = fix_len(a, 24)
    b = fix_len(b, 24)

    print("{}{}{}".format(name, a, b))


def compare(device, target='FP16'):
    fp32_bench_res = load_json_data("fp32_bench_res.json")
    amp_bench_res = load_json_data("amp_bench_res.json")
    latencys = [fp32_bench_res['latency'], \
        amp_bench_res['latency']]
    throughputs = [fp32_bench_res['throughputs'], \
        amp_bench_res['throughputs']]   

    target_label = "{}".format(target)
    format_print('Model', ['FP32', target_label])
    format_print('Latency (s)', latencys)
    format_print('Throughputs (FPS) BS=128', throughputs)

    latencys_times = [1, latencys[1]/latencys[0]]
    throughputs_times = [1, throughputs[1]/throughputs[0]]


    format_print('Model', ['FP32', target_label])
    format_print('Latency Normalized', latencys_times)
    format_print('Throughputs Normalized', throughputs_times)


def help(me):
    print("{} bf16|fp16")

def main(amp):
    warmup_ratio = 0.2
    compare(warmup_ratio, target=amp)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Miss parameters!")
        help(sys.argv[0])
        sys.exit(1)
    me = sys.argv[0]
    amp = sys.argv[1]
    if amp not in ['bf16', 'fp16']:
        print("Parameter value is wrong!")
        help(me)
        sys.exit(1)
    amp = amp.upper()
    main(amp)
