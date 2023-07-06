# Graph fusion

Intel® Extension for TensorFlow\* provides graph optimization to fuse specified operator patterns into a new single operator for better performance.

## Basic fusion

The basic list of supported fusions is shown below. These fusions require input and output of the same data type.

| Pattern | Operator number |
| -- | -- |
| (`Equal`, `NotEqual`, `GreaterEqual`, `Greater`, `LessEqual`, `Less`)+Cast | 2 |
| `L2loss`+`AddN` | 2 |
| `BatchMatMul`+`Mul` | 2 |
| `Mul`+`AddN`+`TrainingOp` | 3 |
| `Conv`+`Bias` | 2 |
| `Conv`+`Bias`+(`Relu`, `Relu6`, `Elu`, `LeakyRelu`, `Gelu_erf`, `Gelu_tanh`, `Tanh`, `Sigmoid`) | 3 |
| `MatMul`+`Bias` | 2 |
| `MatMul`+`Bias`+(`Relu`, `Relu6`, `Elu`, `Gelu_erf`, `Gelu_tanh`, `Tanh`, `Sigmoid`) | 3 |
| `FusedBatchNorm+Relu` | 2 |
| `FusedBatchNormGrad+ReluGrad` | 2 |
| `Conv+Bias+Add` | 3 |
| `Conv`+`Bias`+`Add`+(`Relu`, `Relu6`, `Elu`, `LeakyRelu`, `Gelu_erf`, `Gelu_tanh`, `Tanh`, `Sigmoid`) | 4 |
| `MatMul`+`Bias`+`Add` | 3 |
| `MatMul`+`Bias`+`Add`+(`Relu`, `Relu6`, `Elu`,  `Gelu_erf`, `Gelu_tanh`, `Tanh`, `Sigmoid`) | 4 |
| `MatMul+BiasAddGrad` | 2 |
| `ConvGradFilter`+`BiasAddGrad` | 2 |
| `Pad`+`Conv` | 2 |
| `BatchMatMul` with variable post-op | 2+ |
| `Swish` | 2 |
| `LayerNorm` | 3+ |

## Mixed data type fusion

As stock TensorFlow only supports same-data-type input and output, inserting a cast node during `BF16` inference and training may break the existing fusion pattern and impact performance.

Intel® Extension for TensorFlow\* provides mixed data type fusion, which removes the additional data type conversions on the graph level.

Here is the list of supported mixed data type fusions, and we'll take a closer look at `MatMul` as an example.

| Pattern | Fused operator | Input data type | Output data type| oneDNN `FP32` Math mode |
| --      | --        | -- | -- | -- |
| `MatMul + Cast` | `AccMatMul` | `BF16` | `FP32` | N/A |
| `FusedMatMul + Cast` | `FusedAccMatMul` | `BF16` | `FP32` | N/A |
| `AccMatMul + any MatMul Fusion` | `FusedAccMatMul` | `BF16` | `FP32` | N/A |
| `Cast + MatMul + Cast` | `AccMatMul` | `FP32` | `FP32` | `BF16` |
| `Cast + FusedMatMul + Cast` | `FusedAccMatMul` | `FP32` | `FP32` | `BF16` |

#### Implementation Details

The `Cast + (Fused)MatMul + Cast` pattern is covered by pattern matcher; the rest is covered by remapper fusion.
The new kernels are implemented（`AccMatMul` and `FusedAccMatMul(WithSum)`）as an extension of original `MatMul` with the following new attributes:

- `Tout`: Output data type ∈ {`float32`}.
- `Tpost`: Post op data type ∈ {`bfloat16`, `float32`}.
- `is_bf16_math_mode`: A Boolean to indicate whether to use oneDNN `BF16` math mode if `FP32` input, `FP32` output.

## Generic layout optimizer

As the channels_first format is not supported by stock TensorFlow on CPU, it inserts transpose nodes before and after the Conv3D/MaxPool3D nodes. However, this problem does not exist in GPU device. To avoid unnecessary layout transformation when running on a GPU device, Intel® Extension for TensorFlow\* adds a separate layout optimizer.

| Pattern | Fused operator | Conv data format (before optimization)  | Conv data format (after optimization)| 
| --      | --        | -- | -- | 
| `Transpose + Conv3D + Transpose` | `Conv3D` | `NDHWC` | `NCDHW` |
