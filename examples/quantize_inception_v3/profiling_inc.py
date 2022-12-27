
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


import tensorflow as tf
print("Tensorflow version {}".format(tf.__version__))
tf.config.run_functions_eagerly(False)

from tensorflow.core.protobuf import rewriter_config_pb2

infer_config = tf.compat.v1.ConfigProto()
infer_config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF

session = tf.compat.v1.Session(config=infer_config)
tf.compat.v1.keras.backend.set_session(session)
    
import numpy as np
import time
import argparse
import os
import json
import math

image_size = (224, 224)
batch_size = 1

def process(image,label):
    image = tf.cast(image/255.0 ,tf.float32)
    return image,label

def load_dataset():
    dataset_folder = 'flower_photos/'
    
    train_dataset, val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_folder,
        validation_split=0.2,
        subset="both",
        seed=100,
        image_size=image_size,
        batch_size=batch_size
     )

    class_names = train_dataset.class_names
    class_num = len(class_names)
    
    train_dataset = train_dataset.map(process)
    val_dataset = val_dataset.map(process)
    
    ks = []
    vs = []
    for k, v in list(train_dataset):
        ks.append(k)
        vs.append(v)
    
    return [ks, vs]


def calc_accuracy(predictions, labels):  
    predictions = np.argmax(predictions.numpy(), axis=1)
    same = 0
    for i, x in enumerate(predictions):
        if x == labels[i]:
            same += 1
    if len(predictions) == 0:
        return 0
    else:
        return same / len(predictions)

def rm_dim_1(tensor):
    shape = list(tensor.shape)
    new_shape=[shape[0]]
    new_shape.extend(shape[2:])
    tensor = tf.reshape(tensor, new_shape)

    return tensor

def run_bs(infer, inputs, bs):
    if len(inputs)<bs:
        bt = time.time()
        tens = tf.convert_to_tensor(inputs)
        tens = rm_dim_1(tens)
        infer(tens)
        et = time.time()
        return (et-bt), len(inputs)
    times = math.floor(len(inputs)/bs)
    
    run_time = 0.0
    for i in range(times):
        input_tensor = tf.convert_to_tensor(inputs[i*bs:(i+1)*bs])
        input_tensor = rm_dim_1(input_tensor)
        bt = time.time()
        infer(input_tensor)
        run_time += (time.time()-bt)
        
    return run_time/times, bs

def test_perf(pb_model_file, val_data):
    [x_test_np, label_test] = val_data
    q_model = tf.saved_model.load(pb_model_file)
    x_test = tf.convert_to_tensor(x_test_np)
    infer = q_model.signatures["serving_default"]
    x_test = rm_dim_1(x_test)

    res = infer(x_test)
    res = list(res.values())[0]    
    accuracy = calc_accuracy(res, label_test)    
    print('accuracy:', accuracy)
    
    times = 1
    bs=1024
    warmup = int(times*0.2)
    for i in range(warmup):
        res = run_bs(infer, x_test_np, bs)        
    
    run_time_res = []
    for i in range(times-warmup):
        run_time, cur_bs = run_bs(infer, x_test_np, bs)        
        run_time_res.append(run_time)
    
    avg_run_time = sum(run_time_res)/len(run_time_res)
    throughput = cur_bs*(times-warmup)/avg_run_time
    print('bs:', cur_bs)
    print('max throughput(fps):', throughput)
    
    # latency when BS=1
    times = 1

    bt = 0
    warmup = int(times*0.2)
    sample_num = 200 if len(x_test_np)>200 else len(x_test_np)
    
    for i in range(times):
        if i == warmup:
            bt = time.time()
        for i in range(sample_num):
            tens = tf.convert_to_tensor([x_test_np[i]])
            tens = rm_dim_1(tens)
            #print("zjy4", tens.shape)
            res = infer(tens)            
        
    et = time.time()

    latency = (et - bt) * 1000 / (times - warmup)/sample_num
    print('latency(ms):', latency)

    return accuracy, throughput, latency


def save_res(result):
    accuracy, throughput, latency = result
    res = {}
    res['accuracy'] = accuracy
    res['throughput'] = throughput
    res['latency'] = latency

    outfile = args.index + ".json"
    with open(outfile, 'w') as f:
        json.dump(res, f)
        print("Save result to {}".format(outfile))

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=str, help='file name of output', required=True)

parser.add_argument('--input-graph', type=str, help='file name for graph', required=True)

parser.add_argument('--num-intra-threads', type=str, help='number of threads for an operator', required=False,
                    default="24" )
parser.add_argument('--num-inter-threads', type=str, help='number of threads across operators', required=False,
                    default="1")
parser.add_argument('--omp-num-threads', type=str, help='number of threads to use', required=False,
                    default="24")

args = parser.parse_args()
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
os.environ["TF_NUM_INTEROP_THREADS"] = args.num_inter_threads
os.environ["TF_NUM_INTRAOP_THREADS"] = args.num_intra_threads
#os.environ["DNNL_VERBOSE"] = "1"
dataset = load_dataset()
save_res(test_perf(args.input_graph, dataset))
