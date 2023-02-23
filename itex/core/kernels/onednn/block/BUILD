load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

# List all kernels here.
itex_xpu_library(
    name = "block_kernels",
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":aggregate_ops",
        ":batch_matmul_op",
        ":cast_op",
        ":concat_op",
        ":conv_ops",
        ":dequantize_op",
        ":dequantize_reshape_op",
        ":eltwise_op",
        ":fused_batch_norm_op",
        ":identity_op",
        ":instance_norm_op",
        ":layer_norm_op",
        ":matmul_op",
        ":onednn_to_tf_op",
        ":pool_ops",
        ":quantize_op",
        ":quantized_depthwise_conv_op",
        ":quantized_maxpool_op",
        ":requantization_range_per_channel_op",
        ":requantize_per_channel_op",
        ":reshape_op",
        ":resize_bilinear_op",
        ":resize_nearest_neighbor_op",
        ":shape_op",
        ":slice_op",
        ":softmax_op",
        ":transpose_op",
    ] + select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            ":cwise_ops_common",
        ],
        "//conditions:default": [],
    }),
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
    name = "batch_matmul_op",
    srcs = ["batch_matmul_op.cc"],
    hdrs = [
        "matmul_op.h",
        "//itex/core/kernels/common:matmul_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
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
    name = "cast_op",
    srcs = ["cast_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "concat_op",
    srcs = ["concat_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:noop_hdrs",
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
    name = "conv_ops",
    srcs = [
        "conv_grad_filter_ops.cc",
        "conv_grad_input_ops.cc",
        "conv_ops_impl_cpu.cc",
        "conv_ops_impl_gpu.cc",
    ],
    hdrs = [
        "conv_ops_impl.h",
        "quantized_ops.h",
        "//itex/core/kernels/common:cast_hdrs",
        "//itex/core/kernels/common:conv_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
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
    name = "cwise_ops_common",
    srcs = ["cwise_ops_common.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:cwise_ops_lib",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "dequantize_op",
    srcs = ["dequantize_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:host_data_cache_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
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
    name = "shape_op",
    srcs = ["shape_op.cc"],
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
    name = "dequantize_reshape_op",
    srcs = ["dequantize_reshape_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:host_data_cache_hdrs",
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
    name = "eltwise_op",
    srcs = ["eltwise_op.cc"],
    hdrs = ["eltwise_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fused_batch_norm_op",
    srcs = ["fused_batch_norm_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/common:fused_batch_norm_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "identity_op",
    srcs = ["identity_op.cc"],
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
    name = "instance_norm_op",
    srcs = ["instance_norm_op.cc"],
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
    name = "layer_norm_op",
    srcs = ["layer_norm_op.cc"],
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
    name = "matmul_op",
    srcs = ["matmul_op.cc"],
    hdrs = ["matmul_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "onednn_to_tf_op",
    srcs = ["onednn_to_tf_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "pool_ops",
    srcs = [
        "pool_op.cc",
    ],
    hdrs = [
        "//itex/core/kernels/common:noop_hdrs",
        "//itex/core/kernels/common:pooling_common_hdrs",
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
    srcs = ["quantize_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:host_data_cache_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
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
    name = "requantization_range_per_channel_op",
    srcs = ["requantization_range_per_channel_op.cc"],
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
    name = "requantize_per_channel_op",
    srcs = ["requantize_per_channel_op.cc"],
    hdrs = ["//itex/core/kernels/common:host_data_cache_hdrs"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "reshape_op",
    srcs = ["reshape_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "resize_bilinear_op",
    srcs = ["resize_bilinear_op.cc"],
    hdrs = ["resize_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "resize_nearest_neighbor_op",
    srcs = ["resize_nearest_neighbor_op.cc"],
    hdrs = ["resize_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
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
    name = "softmax_op",
    srcs = ["softmax_op.cc"],
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
    hdrs = [
        "//itex/core/kernels/common:noop_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:transpose_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "quantized_maxpool_op",
    srcs = ["quantized_maxpool_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:noop_hdrs",
        "//itex/core/kernels/common:pooling_common_hdrs",
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
    name = "quantized_depthwise_conv_op",
    srcs = ["quantized_depthwise_conv_op.cc"],
    hdrs = [
        "conv_ops_impl.h",
        "quantized_ops.h",
        "//itex/core/kernels/common:cast_hdrs",
        "//itex/core/kernels/common:conv_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

filegroup(
    name = "quantized_hdrs",
    srcs = [
        "quantized_ops.h",
    ],
    visibility = ["//visibility:public"],
)
