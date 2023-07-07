load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

package(
    licenses = ["notice"],  # Apache 2.0
)

itex_xpu_library(
    name = "conv_ops",
    srcs = [
        "conv_grad_filter_ops.cc",
        "conv_grad_input_ops.cc",
        "conv_ops.cc",
    ],
    hdrs = [
        "//itex/core/kernels/common:conv_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "pooling_ops",
    srcs = [
        "avgpooling_op.cc",
        "maxpooling_op.cc",
    ],
    hdrs = [
        "//itex/core/kernels/common:pooling_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = ["//itex:core"],
    alwayslink = True,
)

itex_xpu_library(
    name = "fused_batch_norm_op",
    srcs = ["fused_batch_norm_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:fused_batch_norm_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fused_batch_norm_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fused_binary_op",
    srcs = ["fused_binary_op.cc"],
    hdrs = [
        "fused_binary_op.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:cwise_ops_lib",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "matmul_op",
    srcs = ["matmul_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:matmul_hdrs",
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
    name = "batch_matmul_op",
    srcs = ["batch_matmul_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:batch_matmul_hdrs",
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
    name = "quantize_op",
    srcs = [
        "quantize_op.cc",
    ],
    hdrs = [
        "//itex/core/kernels/common:quantize_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "dequantize_op",
    srcs = [
        "dequantize_op.cc",
    ],
    hdrs = [
        "//itex/core/kernels/common:dequantize_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "binary_op",
    srcs = [
        "cwise_op_comparison_cast.cc",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:cwise_ops_lib",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "relu_op",
    srcs = ["relu_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:eltwise_base_hdrs",
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
    name = "aggregate_ops",
    srcs = ["aggregate_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/gpu:aggregate_hdrs",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "softmax_op",
    srcs = ["softmax_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:softmax_hdrs",
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
    name = "random_op",
    srcs = ["random_op.cc"],
    hdrs = [
        "random_op_cpu.h",
        "//itex/core/kernels/common:random_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/utils/lib/random:guarded_philox_random",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fused_random_op",
    srcs = ["fused_random_op.cc"],
    hdrs = [
        "random_op_cpu.h",
        "//itex/core/kernels/common:random_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/utils/lib/random:guarded_philox_random",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "resize_bilinear_op",
    srcs = ["resize_bilinear_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "transpose_op",
    srcs = ["transpose_op.cc"],
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
    name = "layer_norm_ops",
    srcs = ["layer_norm_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:layer_norm_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "instance_norm_ops",
    srcs = ["instance_norm_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:instance_norm_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "quantized_matmul",
    srcs = ["quantized_matmul.cc"],
    hdrs = [
        "//itex/core/kernels/common:cast_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
        "//itex/core/kernels/common:quantized_matmul_common_hdrs",
        "//itex/core/kernels/onednn/block:quantized_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "quantized_conv",
    srcs = ["quantized_conv_ops.cc"],
    hdrs = [
        "//itex/core/kernels/common:cast_hdrs",
        "//itex/core/kernels/common:conv_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
        "//itex/core/kernels/common:quantized_conv_hdrs",
        "//itex/core/kernels/onednn/block:quantized_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "quantized_reshape_op",
    srcs = [
        "quantized_reshape_op.cc",
    ],
    hdrs = [
        "//itex/core/kernels/common:quantized_reshape_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "slice_op",
    srcs = ["slice_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:slice_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "gru_ops",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:gru_ops",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "quantized_concat_op",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:quantized_concat_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "einsum_op",
    srcs = ["einsum_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:einsum_hdrs",
        "//itex/core/kernels/common:host_data_cache_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

CPU_KERNELS = [
    ":aggregate_ops",
    ":binary_op",
    ":batch_matmul_op",
    ":conv_ops",
    ":dequantize_op",
    ":einsum_op",
    ":fused_batch_norm_op",
    ":fused_binary_op",
    ":fused_random_op",
    ":gru_ops",
    ":instance_norm_ops",
    ":layer_norm_ops",
    ":matmul_op",
    ":pooling_ops",
    ":quantize_op",
    ":quantized_concat_op",
    ":quantized_conv",
    ":quantized_matmul",
    ":quantized_reshape_op",
    ":random_op",
    ":relu_op",
    ":resize_bilinear_op",
    ":slice_op",
    ":softmax_op",
    ":transpose_op",
]

itex_xpu_library(
    name = "cpu_kernel_impl",
    srcs = ["cpu_kernel_init.cc"],
    hdrs = [
        "cpu_kernel_init.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/legacy:legacy_kernels",
        "//itex/core/kernels/onednn:onednn_kernels",
    ] + select({
        "//third_party/onednn:build_with_onednn_graph": ["//itex/core/kernels/onednn_graph:onednn_graph_op"],
        "//conditions:default": [],
    }) + CPU_KERNELS,
    alwayslink = True,
)
