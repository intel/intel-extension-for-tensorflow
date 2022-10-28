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

import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
import time
import json
import os

def load_pb(pb_file):
    with open(pb_file, 'rb') as f:
        gd = GraphDef()
        gd.ParseFromString(f.read())
    return gd

def get_concrete_function(graph_def, inputs, outputs, print_graph=False):
    def imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrap_function = tf.compat.v1.wrap_function(imports_graph_def, [])
    graph = wrap_function.graph

    return wrap_function.prune(
        tf.nest.map_structure(graph.as_graph_element, inputs),
        tf.nest.map_structure(graph.as_graph_element, outputs))
        
def save_json_data(logs, filename="train.json"):
    with open(filename,"w") as f:
        json.dump(logs,f)
   

def run_infer(concrete_function, shape):
    total_times = 50
    warm = int(0.2*total_times)
    res = []
    for i in range(total_times):
        input_x = tf.random.uniform(shape, minval = 0, maxval= 1.0, dtype=tf.float32)
        bt = time.time()        
        y=concrete_function(input=input_x)
        delta_time = time.time() - bt
        print('Iteration %d: %.3f sec' % (i, delta_time))
        if i >= warm:
            res.append(delta_time)
    latency = sum(res) / len(res)
    return latency
    
def do_benchmark(pb_file, inputs, outputs, base_shape):

    concrete_function = get_concrete_function(
        graph_def=load_pb(pb_file),
        inputs=inputs,
        outputs=outputs,
        print_graph=True)
    bs = 1
    base_shape.insert(0, bs)
    shape = tuple(base_shape)
    latency = run_infer(concrete_function, shape)
    res = {'latency':latency}
    
    bs = 128
    base_shape.insert(0, bs)
    shape = tuple(base_shape)
    latency = run_infer(concrete_function, shape)
    res['throughputs'] = shape[0]/latency    
    
    bench_res_file = "bench_res.json"
    save_json_data(res, bench_res_file)

    print("Benchmark is done!")
    print("Benchmark res {}".format(res))
    print("Finished")
    return res
    
def benchmark():
    pb_file = "inceptionv4_fp32_pretrained_model.pb"
    inputs = ['input:0']
    outputs = ['InceptionV4/Logits/Predictions:0']
    base_shape = [299, 299, 3]

    return do_benchmark(pb_file, inputs, outputs, base_shape)
if __name__ == "__main__":
    benchmark()
