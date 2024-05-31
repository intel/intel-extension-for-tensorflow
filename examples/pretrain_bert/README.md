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
cp ../../../../patch . # When applying this patch, please move it to the above BERT dir first.
git am --signoff < patch  # It's ok when meeting warning about whitespace errors.
```

### Prepare for GPU

Refer to [Prepare](../common_guide_running.md#prepare).

### Setup Running Environment


* Setup for GPU

```bash
./pip_set_env.sh
```
If you use conda env, you can install packages in `./pip_set_env.sh` manually.

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.md#running)

### Prepare Dataset

Nvidia-bert repository provides scripts to download, verify, and extract the SQuAD dataset and pretrained weights for fine-tuning as well as Wikipedia and BookCorpus dataset for pre-training. 

You can run below scripts to download datasets for fine-tuning and pretraining. Assume current_dir is `examples/pretrain_bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT`. And you should modify the environment varible `BERT_PREP_WORKING_DIR` in `data/create_datasets_from_start.sh` to the absolute/relative path of `examples/pretrain_bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT/data`.

```
bash data/create_datasets_from_start.sh all wiki_only
```

For more details about downloading and processing the dataset, you can reference [downloading](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT#quick-start-guide) and [processing](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT#getting-the-data) part. After downloading and processing, the datasets are supposed in the following locations by default

- SQuAD v1.1 - `data/download/squad/v1.1`
- SQuAD v2.0 - `data/download/squad/v2.0`
- BERT-Large - `data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16`
- BERT-Base - `data/download/google_pretrained_weights/uncased_L-12_H-768_A-12`
- Wikipedia TFRecords - `data/tfrecords/wikicorpus_en`

## Execute the Example

Bert pretraining is very time-consuming, as nvidia-bert repository says, training BERT-Large from scratch on 16 V100 using FP16 datatype takes around 4.5 days. So Here we only provide single-tile pretraining scripts within a day to show performance.

### Prerequisites
We use `intel-optimization-for-horovod` to implement efficient multi-GPU training with OneCCL. If you want to use multi-GPU, please replace `horovod` with `intel-optimization-for-horovod`.
```
pip uninstall horovod
pip install intel-optimization-for-horovod
```

### Pretraining Command

Assume current_dir is `examples/pretrain_bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT`. Please set DATA_DIR to your real data dir.

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

+ BFloat16 DataType

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

+ Float32/TF32 DataType
To avoid Out-Of-Memory, please use below hyperparameters instead of above bf16 ones.

```
TRAIN_BATCH_SIZE_PHASE1=60
TRAIN_BATCH_SIZE_PHASE2=10
LEARNING_RATE_PHASE1=7.5e-4
LEARNING_RATE_PHASE2=5e-4
NUM_ACCUMULATION_STEPS_PHASE1=64
NUM_ACCUMULATION_STEPS_PHASE2=192
```

**Note**: If you want to run on more cards and even multi-node, you could adjust the parameter `NUM_GPUS`.

### Finetune Command

Assume current_dir is `examples/pretrain_bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT`. `$PRETRAIN_RESULT_DIR` is from previous pretraining. It will finetune using this pretraining checkpoint.

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
BATCH_SIZE_PER_GPU=76
LEARNING_RATE_PER_GPU=3e-5
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

**Note**: If you want to run on more cards and even multi-node, you could adjust the parameter `NUM_GPUS`. 
+ If you meet Out-Of-Memory error, you can reduce batch_size and use below hyperparameters.
```
BATCH_SIZE_PER_GPU=12
LEARNING_RATE_PER_GPU=5e-6
```

## Convergence
If you want to run Bert-Large pretraining to convergence, we provide a set of bfloat16 distributed training hyperparameters verified on 4 cards(8 tiles) and 12 cards(24 tiles) PVC.

### Prerequisites
We use `intel-optimization-for-horovod` to implement efficient multi-GPU training with OneCCL. If you want to use multi-GPU, please replace `horovod` with `intel-optimization-for-horovod`.
```
pip uninstall horovod
pip install intel-optimization-for-horovod
```

### Commands

Assume current_dir is `examples/pretrain_bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT`

- Pretraining
```
TRAIN_BATCH_SIZE_PHASE1=312
TRAIN_BATCH_SIZE_PHASE2=40
EVAL_BATCH_SIZE=8
LEARNING_RATE_PHASE1=0.004888
LEARNING_RATE_PHASE2=0.004
DATATYPE=bf16
USE_XLA=false
NUM_GPUS=8
WARMUP_STEPS_PHASE1=2000
WARMUP_STEPS_PHASE2=200
TRAIN_STEPS=6416
SAVE_CHECKPOINT_STEPS=100
NUM_ACCUMULATION_STEPS_PHASE1=256
NUM_ACCUMULATION_STEPS_PHASE2=768
BERT_MODEL=large

LEARNING_RATE_PHASE1=$(echo $LEARNING_RATE_PHASE1 $NUM_GPUS | awk '{printf ("%0.7f", $1/$2)}')
LEARNING_RATE_PHASE2=$(echo $LEARNING_RATE_PHASE2 $NUM_GPUS | awk '{printf ("%0.7f", $1/$2)}')
NUM_ACCUMULATION_STEPS_PHASE1=$(echo $NUM_ACCUMULATION_STEPS_PHASE1 $NUM_GPUS | awk '{print int($1/$2)}')
NUM_ACCUMULATION_STEPS_PHASE2=$(echo $NUM_ACCUMULATION_STEPS_PHASE2 $NUM_GPUS | awk '{print int($1/$2)}')
GBS1=$(expr $TRAIN_BATCH_SIZE_PHASE1 \* $NUM_GPUS \* $NUM_ACCUMULATION_STEPS_PHASE1)
GBS2=$(expr $TRAIN_BATCH_SIZE_PHASE2 \* $NUM_GPUS \* $NUM_ACCUMULATION_STEPS_PHASE2)

PRETRAIN_RESULT_DIR=./results/tf_bert_pretraining_lamb_${BERT_MODEL}_${DATATYPE}_gbs1_${GBS1}_gbs2_${GBS2}
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

**Note**: If you want to run on more cards and even multi-node, you could adjust the parameter `NUM_GPUS`.

- Finetune
```
NUM_GPUS=8
BATCH_SIZE_PER_GPU=12
LEARNING_RATE_PER_GPU=5e-6
DATATYPE=bf16
USE_XLA=false
SQUAD_VERSION=1.1
EPOCHS=2
USE_MYTRAIN=true
BERT_MODEL=large

PRETRAIN_PATH=$PRETRAIN_RESULT_DIR/phase_2/pretrained/bert_model.ckpt-1
DATA_DIR=$DATA_DIR
RESULT_DIR=./results/tf_bert_finetune_${BERT_MODEL}_${DATATYPE}

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

**Note**: If you want to run on more cards and even multi-node, you could adjust the parameter `NUM_GPUS`.

### Results
- Pretraining Phase1
```
[0] I1015 21:28:32.105446 22471978829632 model_training_utils.py:83] Training Summary: 
[0] {'total_training_steps': 5774, 'train_loss': xxxxx}
[0] I1015 21:28:32.190052 22471978829632 model_training_utils.py:595] -----------------------------
[0] I1015 21:28:32.190136 22471978829632 model_training_utils.py:596]   Batch size = 312
[0] I1015 21:28:32.190169 22471978829632 model_training_utils.py:597]   Num steps = 5774
[0] I1015 21:28:32.190193 22471978829632 model_training_utils.py:598]   LR = 0.000611
[0] I1015 21:28:32.190217 22471978829632 model_training_utils.py:600] Multi-GPU training with TF Horovod
[0] I1015 21:28:32.190249 22471978829632 model_training_utils.py:601] hvd.size() = 8
[0] I1015 21:28:32.190273 22471978829632 model_training_utils.py:602] Total Training Time = xxxxx for Sequences = 461180928
[0] I1015 21:28:32.190343 22471978829632 model_training_utils.py:604] Throughput Average (sequences/sec) with overhead = xxxxx
[0] I1015 21:28:32.190536 22471978829632 model_training_utils.py:606] Throughput Average (sequences/sec) = xxxxx
[0] I1015 21:28:32.190559 22471978829632 model_training_utils.py:607] -----------------------------
[0] decayed_learning_rate_at_crossover_point = 4.888000e-03, adjusted_init_lr = 4.888000e-03
[0] DLL 2023-10-15 21:28:32.190587 -  throughput_train : xxxxx sequences/s
[0] DLL 2023-10-15 21:28:32.190668 -  total_loss : xxxxx
```
- Pretraining Phase2
```
[0] I1021 03:48:52.227250 23092487755584 model_training_utils.py:83] Training Summary: 
[0] {'total_training_steps': 1666, 'train_loss': xxxxx}
[0] I1021 03:48:52.314010 23092487755584 model_training_utils.py:595] -----------------------------
[0] I1021 03:48:52.314064 23092487755584 model_training_utils.py:596]   Batch size = 40
[0] I1021 03:48:52.314090 23092487755584 model_training_utils.py:597]   Num steps = 1666
[0] I1021 03:48:52.314113 23092487755584 model_training_utils.py:598]   LR = 0.0005
[0] I1021 03:48:52.314135 23092487755584 model_training_utils.py:600] Multi-GPU training with TF Horovod
[0] I1021 03:48:52.314165 23092487755584 model_training_utils.py:601] hvd.size() = 8
[0] I1021 03:48:52.314197 23092487755584 model_training_utils.py:602] Total Training Time = xxxxx for Sequences = 51179520
[0] I1021 03:48:52.314260 23092487755584 model_training_utils.py:604] Throughput Average (sequences/sec) with overhead = xxxxx
[0] I1021 03:48:52.314455 23092487755584 model_training_utils.py:606] Throughput Average (sequences/sec) = xxxxx
[0] I1021 03:48:52.314478 23092487755584 model_training_utils.py:607] -----------------------------
[0] decayed_learning_rate_at_crossover_point = 4.000000e-03, adjusted_init_lr = 4.000000e-03
[0] DLL 2023-10-21 03:48:52.314505 -  throughput_train : xxxxx sequences/s
[0] DLL 2023-10-21 03:48:52.314589 -  total_loss : xxxxx
```
- Finetune
```
[0] I1022 18:28:51.037595 23450678511424 model_training_utils.py:83] Training Summary: 
[0] {'total_training_steps': 1846, 'train_loss': xxxxx}
[0] I1022 18:28:51.072701 23450678511424 model_training_utils.py:595] -----------------------------
[0] I1022 18:28:51.072752 23450678511424 model_training_utils.py:596]   Batch size = 12
[0] I1022 18:28:51.072779 23450678511424 model_training_utils.py:597]   Num steps = 1846
[0] I1022 18:28:51.072805 23450678511424 model_training_utils.py:598]   LR = 5e-06
[0] I1022 18:28:51.072829 23450678511424 model_training_utils.py:600] Multi-GPU training with TF Horovod
[0] I1022 18:28:51.072861 23450678511424 model_training_utils.py:601] hvd.size() = 8
[0] I1022 18:28:51.072884 23450678511424 model_training_utils.py:602] Total Training Time = xxxxx for Sequences = 177216
[0] I1022 18:28:51.072928 23450678511424 model_training_utils.py:604] Throughput Average (sequences/sec) with overhead = xxxxx
[0] I1022 18:28:51.073072 23450678511424 model_training_utils.py:606] Throughput Average (sequences/sec) = xxxxx
[0] I1022 18:28:51.073094 23450678511424 model_training_utils.py:607] -----------------------------
```

- Inference after finetune
```
[0] DLL 2023-10-22 18:28:51.073121 -  throughput_train : xxxxx sequences/s
[0] DLL 2023-10-22 18:28:51.073183 -  total_loss : xxxxx
[0] DLL 2023-10-22 18:30:21.122414 -  f1 : xxxxx None
[0] DLL 2023-10-22 18:30:21.122509 -  exact_match : xxxxx 
[0] b'{"exact_match": xxxxx, "f1": xxxxx}\n'
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.

``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
2. If you get the following error log, please uninstall current horovod using `pip uninstall horovod`. And reinstall horovod using `pip install --no-cache-dir horovod`.

```
horovod.common.exceptions.HorovodVersionMismatchError: Framework tensorflow installed with version 2.15.1 but found version 2.14.1.
             This can result in unexpected behavior including runtime errors.
             Reinstall Horovod using `pip install --no-cache-dir` to build with the new version.
```

3. If you get Out-Of-Memory error log using above pretraining scripts, you can reduce batch_size to avoid it.