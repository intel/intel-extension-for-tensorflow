# FP8 BERT-Large Fine-tuning for Classifying Text on Intel GPU

## Introduction

Intel® Extension for TensorFlow* is compatible with stock TensorFlow*. 
This example shows FP8 BERT-Large Fine-tuning for Classifying Text.

Install the Intel® Extension for TensorFlow* in legacy running environment, Tensorflow will execute the Training on Intel GPU.

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Max Series
 - Intel® Data Center GPU Flex Series 170
 
## Prerequisites

### Model Code change
We set up FP8 BERT-Large fine-tuning based on official google-bert. If you want to check the accuracy of FP8 BERT-Large fine-tuning, please clone google-bert and apply patch:
```
git clone https://github.com/google-research/bert.git
cd bert
git apply patch
```

### Prepare for GPU

Refer to [Prepare](../common_guide_running.html#prepare)

### Setup Running Environment


* Setup for GPU
```bash
./pip_set_env.sh
```

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.html#running)


### Execute the Example
#### BF16 + FP8 Fine-tuning
```
export BERT_LARGE_DIR=/path/to/bert-large-dir
export SQUAD_DIR=/path/to/squad-dir
export OUTPUT_DIR=/path/to/output-dir
python run_squad.py \
   --vocab_file=$BERT_LARGE_DIR/vocab.txt \
   --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
   --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
   --do_train=True \
   --train_file=$SQUAD_DIR/train-v1.1.json \
   --do_predict=True \
   --predict_file=$SQUAD_DIR/dev-v1.1.json \
   --train_batch_size=32 \
   --learning_rate=3e-5 \
   --num_train_epochs=2.0 \
   --max_seq_length=512 \
   --doc_stride=128 \
   --output_dir=$OUTPUT_DIR \
   --use_tpu=False \
   --precision=bfloat16 \
```

#### Accuracy
The dev set predictions will be saved into a file called predictions.json in the output_dir using the above command,

```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $OUTPUT_DIR/predictions.json
```
Which should produce an output like this:
```
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.
``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
