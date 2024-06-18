import os
import tensorflow as tf
import argparse
import json
import time
import keras
import keras_nlp
import numpy as np
import kagglehub

# Download latest version
#path = kagglehub.model_download("keras/llama3/keras/llama3_8b_en")
#print("Path to model files:", path)

parser = argparse.ArgumentParser()
parser.add_argument(
  "--model",
  type=str,
  choices=["llama3_8b_en", "llama3_instruct_8b_en"],
  default="llama3_8b_en",
  help="the mdoel name",
)
parser.add_argument(
  "--data-dir",
  type=str,
  default="./",
  help="the dataset path",
)
parser.add_argument(
  "--dtype",
  type=str,
  choices=["float32", "bfloat16"],
  default="float32",
  help="float32, bfloat16"
)
parser.add_argument(
  "--prompt", default=None, type=str, help="input prompt for self-defined if needed"
)
parser.add_argument(
  "--input-tokens",
  default=None,
  choices=["32", "64", "128", "256", "512", "1024", "2016", "2017", "2048", "4096", "8192"],
  type=str,
  help="input tokens length if needed from prompt.json",
)
parser.add_argument(
  "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--num-beams", default=1, type=int, help="beam width")
parser.add_argument("--num-iter", default=10, type=int, help="num iter")
parser.add_argument("--num-warmup", default=3, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")

args = parser.parse_args()
path = args.data_dir + args.model
print("Dataset dir is: %s" % path)

if args.dtype == "bfloat16":
  keras.config.set_floatx("bfloat16")
model = keras_nlp.models.Llama3CausalLM.from_preset(path, dtype=args.dtype)
if args.num_beams > 1:
  from keras_nlp.samplers import BeamSampler
  model.compile(sampler=BeamSampler(num_beams=args.num_beams))
else:
  model.compile(sampler="greedy")

if args.prompt is not None:
  prompt = args.prompt
elif args.input_tokens is not None:
  current_path = os.path.dirname(__file__)
  with open(str(current_path) + "/prompt.json") as f:
    prompt_pool = json.load(f)
  prompt = prompt_pool[args.input_tokens]
print("Prompt: %s!" % prompt)

total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
prompt = [prompt] * args.batch_size
for i in range(num_iter):
  tic = time.time()
  output = model.generate(
    prompt, max_length=int(args.max_new_tokens)+int(args.input_tokens)
  )
  toc = time.time()
  print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
  if i >= num_warmup:
    total_time += toc - tic

print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % latency)
print("Output: %s." % output)
