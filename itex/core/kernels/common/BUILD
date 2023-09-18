package(
    licenses = ["notice"],  # Apache 2.0
)

load("//itex:itex.bzl", "cc_library", "itex_xpu_library", "tf_copts")

cc_library(
    name = "common_kernel_hdrs",
    srcs = glob(["*.h"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
    ],
)

filegroup(
    name = "host_data_cache_hdrs",
    srcs = [
        "host_data_cache.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "batch_matmul_hdrs",
    srcs = [
        "batch_matmul_op.h",
        "fill_functor.h",
        "host_data_cache.h",
        "matmul_op.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "control_flow_hdrs",
    srcs = [
        "control_flow_ops.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "conv_hdrs",
    srcs = [
        "conv_grad_ops.h",
        "conv_ops.h",
        "host_data_cache.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "dequantize_hdrs",
    srcs = [
        "dequantize_op.h",
        "host_data_cache.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "eltwise_base_hdrs",
    srcs = [
        "eltwise_base.h",
        "relu_op.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "fused_batch_norm_hdrs",
    srcs = [
        "fused_batch_norm_functor.h",
        "fused_batch_norm_op.h",
        "host_data_cache.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "instance_norm_hdrs",
    srcs = [
        "instance_norm_op.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "matmul_hdrs",
    srcs = [
        "fill_functor.h",
        "host_data_cache.h",
        "matmul_op.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "noop_hdrs",
    srcs = [
        "no_ops.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "pooling_hdrs",
    srcs = [
        "avgpooling_op.h",
        "maxpooling_op.h",
        "pooling_ops_common.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "pooling_common_hdrs",
    srcs = [
        "pooling_ops_common.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "quantize_hdrs",
    srcs = [
        "host_data_cache.h",
        "quantize_op.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "random_hdrs",
    srcs = [
        "random_op.h",
        "random_ops_util.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "softmax_hdrs",
    srcs = [
        "softmax_op.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "layer_norm_hdrs",
    srcs = [
        "layer_norm_op.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "quantized_matmul_common_hdrs",
    srcs = [
        "fill_functor.h",
        "host_data_cache.h",
        "quantized_matmul_common.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "quantized_conv_hdrs",
    srcs = ["quantized_conv_ops.h"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cast_hdrs",
    srcs = ["cast_op.h"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "quantized_reshape_hdrs",
    srcs = [
        "quantized_reshape_op.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "einsum_hdrs",
    srcs = [
        "einsum_op.h",
        "einsum_op_impl.h",
        "fill_functor.h",
        "matmul_op.h",
        "transpose_functor.h",
    ],
    visibility = ["//visibility:public"],
)

itex_xpu_library(
    name = "cwise_ops_lib",
    srcs = ["cwise_ops_common.cc"],
    hdrs = [
        "cwise_ops.h",
        "cwise_ops_common.h",
        "cwise_ops_gradients.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":fill_functor",
        "//itex:core",
    ],
)

itex_xpu_library(
    name = "fill_functor",
    srcs = ["fill_functor.cc"],
    hdrs = ["fill_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fused_batch_norm_functor",
    srcs = ["fused_batch_norm_functor.cc"],
    hdrs = [
        "fused_batch_norm_functor.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":fill_functor",
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "gru_ops",
    srcs = ["gru_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "no_ops",
    srcs = ["no_ops.cc"],
    hdrs = ["no_ops.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "quantized_concat_op",
    srcs = ["quantized_concat_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "slice_functor",
    srcs = ["slice_functor.cc"],
    hdrs = ["slice_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "sparse_math_lib",
    srcs = ["reshape_util.cc"],
    hdrs = ["reshape_util.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "transpose_functor",
    srcs = ["transpose_functor.cc"],
    hdrs = ["transpose_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "transpose_op_lib",
    hdrs = ["transpose_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":transpose_functor",
        "//itex:core",
    ],
    alwayslink = True,
)
