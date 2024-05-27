# Copyright (c) 2021-2023 Intel Corporation
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

import os
import sys


__version__ = '2.15.0.1'

def parse_args(argv):
  result = {}
  for arg in argv:
    k, v = arg.split("=")
    result[k] = v

  return result


def git_hash(header_in):
  with open("bazel-out/volatile-status.txt", 'r') as f:
    for line in f:
      if not line.startswith("ITEX_REVISION"):
        continue
      k, v = line.split(" ")
      return v.strip()

  return "N/A" 

def generate_version(header_in, header_out):
  hash_value = git_hash(header_in)

  [major, minor, patch1, patch2] = __version__.split(".")

  with open(os.path.expanduser(header_in)) as inf:
    content = inf.read()
    content = content.replace("@ITEX_VERSION_MAJOR@", major)
    content = content.replace("@ITEX_VERSION_MINOR@", minor)
    content = content.replace("@ITEX_VERSION_PATCH@", ".".join([patch1, patch2]))
    content = content.replace("@ITEX_VERSION_HASH@", hash_value)

    header_out = os.path.expanduser(header_out)
    header_out_dir = os.path.dirname(header_out)
    if not os.path.exists(header_out_dir):
      os.makedirs(header_out_dir, exist_ok=True)

    with open(header_out, "w") as outf:
      outf.write(content)


def main():
  args = parse_args(sys.argv[1:])
  generate_version(args["--in"], args["--out"])


if __name__ == "__main__":
  main()
