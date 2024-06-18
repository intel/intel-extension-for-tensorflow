# Llama3 Inference Best Known Method for Intel-Extension-For-Tensorflow on Intel GPU

## Introduction
Llama 3 is a collection of pretrained and fine-tuned generative text models ranging in scale from 8 billion to 70 billion parameters. For more detail information, please refer to [llama-3/keras](https://www.kaggle.com/models/metaresearch/llama-3/keras).

This example shows how to run Llama3 8b inference with Intel® Extension for TensorFlow* on Intel GPU.

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Max Series
 - Intel® Data Center GPU Flex Series 170
 
## Prerequisites
### Dataset
Follow [llama-3/keras/llama3_8b_en](https://www.kaggle.com/models/metaresearch/llama-3/keras/llama3_8b_en) to apply access permission and then download datasets.
```
mkdir -p llama3_8b_en
tar -xzvf llama3-keras-llama3_8b_en-v3.tar.gz -C ./llama3_8b_en
```

### Prepare for GPU

Refer to [Prepare](../common_guide_running.md#prepare)

### Setup Running Environment
* Setup for GPU
```bash
./pip_set_env.sh
```
Note: This Llama3 keras3 implementation requires TensorFlow >= 2.16.1 and Intel® Extension for TensorFlow* >= 2.16.0.0.

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.md#running)


### Executes the Example with Python API
#### Model Default Parameters
| **Parameter** | **Default Value** |
| :---: | :--- |
| **model** | llama3_8b_en |
| **dtype** | bfloat16 |
| **data-dir** | ./ |
| **input-tokens** | 32 |
| **max-new-tokens** | 32 |
| **num-beams** | 1 |
| **num-iter** | 10 |
| **num-warmup** | 3 |
| **batch-size** | 1 |

#### FP32 Inference
```
python run_generate.py \
  --model llama3_8b_en \
  --dtype float32      \
  --data-dir ./        \
  --input-tokens 32    \
  --max-new-tokens 32
```

#### BF16 Inference
```
python run_generate.py \
  --model llama3_8b_en \
  --dtype bfloat16     \
  --data-dir ./        \
  --input-tokens 32    \
  --max-new-tokens 32
```

## Example Output
With successful execution, it will print out the following results:

```
Prompt: Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.
Iteration: 0, Time: xxx sec
Iteration: 1, Time: xxx sec
Iteration: 2, Time: xxx sec
Iteration: 3, Time: xxx sec
Iteration: 4, Time: xxx sec
Iteration: 5, Time: xxx sec
Iteration: 6, Time: xxx sec
Iteration: 7, Time: xxx sec
Iteration: 8, Time: xxx sec
Iteration: 9, Time: xxx sec

 ---------- Summary: ----------
Inference latency: xxx sec.
Output: Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun. She wanted to be a princess, and a pirate, and a fairy, and a mermaid, and a superhero, and a witch, and a queen.
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.
``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
