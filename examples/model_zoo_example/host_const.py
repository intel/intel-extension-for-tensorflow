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


import argparse

from utils import read_graph, write_graph
from tensorflow.core.framework import graph_pb2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in_graph", default="", help="Path to input graph")
parser.add_argument("-b", "--input_binary", action="store_true", help="Is binary graph")
parser.add_argument("-o", "--out_graph", default="", help="Path to output graph")
args = parser.parse_args()


def replace_const(graph):
    new_graph = graph_pb2.GraphDef()
    for node in graph.node:
        if node.op == 'Const' and ('min' in node.name or 'max' in node.name):
            node.op = 'HostConst'
        new_graph.node.extend([node])
    return new_graph


def main():
    in_graph = args.in_graph
    is_binary = args.input_binary
    out_graph = args.out_graph

    raw = read_graph(in_graph, is_binary)
    new = replace_const(raw)
    write_graph(new, out_graph)


if __name__ == '__main__':
    main()
