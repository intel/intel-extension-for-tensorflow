# Accelerate BERT-Large Pretraining on Intel GPU

## Introduction

Intel® Extension for TensorFlow* is compatible with stock TensorFlow*. 
This example shows BERT-Large Pretraining.

Install the Intel® Extension for TensorFlow* in legacy running environment, Tensorflow will execute the Training on Intel GPU.

## Hardware Requirements

Verified Hardware Platforms:

 - Intel® Data Center GPU Max Series

## Prerequisites

### Model Code change

We set up BERT-Large pretraining based on nvidia-bert. We optimized nvidia-bert, for example, using custom kernels, fusing some ops to reduce op number, and adding bf16 mode for the model. 

To get better performance, instead of installing official nvidia-bert, you can clone nvidia-bert, apply the patch, then install it as shown here:

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/TensorFlow2/LanguageModeling/BERT
git apply patch  # When applying this patch, please move it to the above BERT dir first.
```

### Prepare for GPU

Refer to [Prepare](../common_guide_running.md#prepare).

### Setup Running Environment


* Setup for GPU

```bash
./pip_set_env.sh
```

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.md#running)

### Prepare Dataset

Nvidia-bert repository provides scripts to download, verify, and extract the SQuAD dataset and pretrained weights for fine-tuning as well as Wikipedia and BookCorpus dataset for pre-training. 

You can run below scripts to download datasets for fine-tuning and pretraining. Assume current_dir is `examples/pretrain_bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT`. And you should modify the environment varible `BERT_PREP_WORKING_DIR` in `data/create_datasets_from_start.sh` to your real data dir. 

```
bash data/create_datasets_from_start.sh all
```

For more details about downloading and processing the dataset, you can reference [downloading](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT#quick-start-guide) and [processing](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT#getting-the-data) part. After downloading and processing, the datasets are supposed in the following locations by default

- SQuAD v1.1 - `data/download/squad/v1.1`
- SQuAD v2.0 - `data/download/squad/v2.0`
- BERT-Large - `data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16`
- BERT-Base - `data/download/google_pretrained_weights/uncased_L-12_H-768_A-12`
- Wikipedia + BookCorpus TFRecords - `data/tfrecords/books_wiki_en_corpus`

## Execute the Example

Bert pretraining is very time-consuming, as nvidia-bert repository says, training BERT-Large from scratch on 16 V100 using FP16 datatype takes around 4.5 days. So Here we only provide single-tile pretraining scripts within a day to show performance.

#### Pretraining Command

Assume current_dir is `examples/pretrain_bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT`

+ BFloat16 DataType

```
DATATYPE=bf16
```

+ Float32 DataType

```
DATATYPE=fp32
```

+ TF32 DataType

```
export ITEX_FP32_MATH_MODE=TF32
DATATYPE=fp32
```

**Final Scripts**

+ We use [LAMB](https://arxiv.org/pdf/1904.00962.pdf) as the optimizer and pretraining has two phases. The maximum sequence length of phase1 and phase2 is 128 and 512, respectively. For the whole process of pretraining, you can use scripts in [nvidia-bert](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT#training-process).

```
TRAIN_BATCH_SIZE_PHASE1=312
TRAIN_BATCH_SIZE_PHASE2=40
EVAL_BATCH_SIZE=8
LEARNING_RATE_PHASE1=8.12e-4
LEARNING_RATE_PHASE2=5e-4
DATATYPE=$DATATYPE
USE_XLA=false
NUM_GPUS=1
WARMUP_STEPS_PHASE1=810
WARMUP_STEPS_PHASE2=81
TRAIN_STEPS=2600
SAVE_CHECKPOINT_STEPS=100
NUM_ACCUMULATION_STEPS_PHASE1=32
NUM_ACCUMULATION_STEPS_PHASE2=96
BERT_MODEL=large

GBS1=$(expr $TRAIN_BATCH_SIZE_PHASE1 \* $NUM_GPUS \* $NUM_ACCUMULATION_STEPS_PHASE1)
GBS2=$(expr $TRAIN_BATCH_SIZE_PHASE2 \* $NUM_GPUS \* $NUM_ACCUMULATION_STEPS_PHASE2)

PRETRAIN_RESULT_DIR=./results/tf_bert_pretraining_lamb_${BERT_MODEL}_${$DATATYPE}_gbs1_${GBS1}_gbs2_${GBS2}
DATA_DIR=$DATA_DIR

bash scripts/run_pretraining_lamb.sh \
    $TRAIN_BATCH_SIZE_PHASE1 \
    $TRAIN_BATCH_SIZE_PHASE2 \
    $EVAL_BATCH_SIZE \
    $LEARNING_RATE_PHASE1 \
    $LEARNING_RATE_PHASE2 \
    $DATATYPE \
    $USE_XLA \
    $NUM_GPUS \
    $WARMUP_STEPS_PHASE1 \
    $WARMUP_STEPS_PHASE2 \
    $TRAIN_STEPS \
    $SAVE_CHECKPOINT_STEPS \
    $NUM_ACCUMULATION_STEPS_PHASE1 \
    $NUM_ACCUMULATION_STEPS_PHASE2 \
    $BERT_MODEL \
    $DATA_DIR \
    $PRETRAIN_RESULT_DIR \
    |& tee pretrain_lamb.log
```

#### Finetune Command

Assume current_dir is `examples/pretrain_bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT`. After getting the pretraining checkpoint, you can use it for finetuning.

+ BFloat16 DataType

```
DATATYPE=bf16
```

+ Float32 DataType

```
DATATYPE=fp32
```

+ TF32 DataType

```
export ITEX_FP32_MATH_MODE=TF32
DATATYPE=fp32
```

**Final Scripts**

```
NUM_GPUS=1
BATCH_SIZE_PER_GPU=12
LEARNING_RATE_PER_GPU=5e-6
DATATYPE=$DATATYPE
USE_XLA=false
SQUAD_VERSION=1.1
EPOCHS=2
USE_MYTRAIN=true
BERT_MODEL=large
PRETRAIN_PATH=$PRETRAIN_RESULT_DIR/phase_2/pretrained/bert_model.ckpt-1
DATA_DIR=$DATA_DIR
RESULT_DIR=./results/tf_bert_finetune_${BERT_MODEL}_${$DATATYPE}
bash scripts/run_squad.sh \
    $NUM_GPUS \
    $BATCH_SIZE_PER_GPU \
    $LEARNING_RATE_PER_GPU \
    $DATATYPE \
    $USE_XLA \
    $BERT_MODEL \
    $SQUAD_VERSION \
    $EPOCHS \
    $USE_MYTRAIN \
    $PRETRAIN_PATH \
    $DATA_DIR \
    $RESULT_DIR \
    |& tee finetune.log
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.

``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
