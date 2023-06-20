load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

itex_xpu_library(
    name = "fp8_quantize_op",
    srcs = ["fp8_quantize_op.cc"],
    hdrs = [
        "fp8_quantize_gpu.h",
        "utils.h",
        "vectorized_pointwise.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fp8_layernorm_op",
    srcs = ["fp8_layernorm_op.cc"],
    hdrs = [
        "fp8_layernorm_gpu.h",
        "utils.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fp8_gelu_op",
    srcs = ["fp8_gelu_op.cc"],
    hdrs = [
        "fp8_gelu_gpu.h",
        "utils.h",
        "vectorized_pointwise.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fp8_quantize_fusion_op",
    srcs = ["fp8_quantize_fusion_op.cc"],
    hdrs = [
        "fp8_quantize_fusion_gpu.h",
        "utils.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fp8_matmul_op",
    srcs = ["fp8_matmul_op.cc"],
    hdrs = [
        "fp8_matmul_gpu.h",
        "fp8_quantize_gpu.h",
        "utils.h",
        "vectorized_pointwise.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fp8_attention_op",
    srcs = ["fp8_attention_op.cc"],
    hdrs = [
        "fp8_attention_bwd_gpu.h",
        "fp8_attention_fwd_gpu.h",
        "fp8_quantize_gpu.h",
        "utils.h",
        "vectorized_pointwise.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:cwise_ops_lib",
        "//itex/core/kernels/common:fill_functor",
        "//itex/core/kernels/gpu:reduction_ops",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fp8_transpose_op",
    srcs = ["fp8_transpose_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:transpose_op_lib",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fp8_op",
    visibility = ["//visibility:public"],
    deps = [
        ":fp8_attention_op",
        ":fp8_gelu_op",
        ":fp8_layernorm_op",
        ":fp8_matmul_op",
        ":fp8_quantize_fusion_op",
        ":fp8_quantize_op",
        ":fp8_transpose_op",
    ],
)
