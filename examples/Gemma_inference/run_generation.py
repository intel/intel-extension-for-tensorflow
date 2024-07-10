import time
import tensorflow as tf
import json
import pathlib
import argparse
import os
import sys
import keras
import keras_nlp

MODEL_CLASSES = {
  "gemma_2b": "gemma_2b_en",
  "gemma_7b": "gemma_7b_en",
  "gemma_2b_it": "gemma_instruct_2b_en",
  "gemma_7b_it": "gemma_instruct_7b_en",
  "gemma2_9b" : "gemma2_9b_en",
  "gemma2_27b" : "gemma2_27b_en",
  "gemma2_9b_it" : "gemma2_instruct_9b_en",
  "gemma2_27b_it" : "gemma2_instruct_27b_en",
}
 
parser = argparse.ArgumentParser()
parser.add_argument(
  "--model",
  type=str,
  choices=["gemma_2b", "gemma_7b", "gemma_2b_it", "gemma_7b_it",
           "gemma2_9b", "gemma2_27b", "gemma2_9b_it", "gemma2_27b_it"],
  default="gemma_2b",
  help="the mdoel name",
)
parser.add_argument(
  "--dtype",
  type=str,
  choices=["float32", "bfloat16"],
  default="float32",
  help="bfloat16, float32",
)
parser.add_argument(
  "--input-tokens",
  default="32",
  choices=["32", "64", "128", "256", "512", "1024", "2016", "2017", "2048", "4096", "8192"],
  type=str,
  help="input tokens length if needed from prompt.json",
)
parser.add_argument(
  "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
  "--prompt", default=None, type=str, help="input prompt for self-defined if needed"
)
parser.add_argument("--num-beams", default=1, type=int, help="beam width")
parser.add_argument("--num-iter", default=10, type=int, help="num iter")
parser.add_argument("--num-warmup", default=3, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
args = parser.parse_args()

if args.dtype == "bfloat16":
  keras.config.set_floatx("bfloat16")
model = keras_nlp.models.GemmaCausalLM.from_preset(MODEL_CLASSES[args.model])
if args.num_beams > 1:
  from keras_nlp.samplers import BeamSampler
  model.compile(sampler=BeamSampler(num_beams=args.num_beams))
current_path = os.path.dirname(__file__)
with open(str(current_path) + "/prompt.json") as f:
  prompt_pool = json.load(f)
prompt = prompt_pool[args.input_tokens]
total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
prompt = [prompt] * args.batch_size
total_list = []
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



