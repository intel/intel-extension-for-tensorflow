load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

package(
    licenses = ["notice"],  # Apache 2.0
)

itex_xpu_library(
    name = "aggregate_ops",
    srcs = ["aggregate_ops.cc"],
    hdrs = [
        "aggregate_ops.h",
        "full_reduction_kernels.h",
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
    name = "aggregate_hdrs",
    hdrs = ["aggregate_ops.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "reduction_utils",
    srcs = ["reduction_utils.cc"],
    hdrs = ["reduction_utils.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "argmax_op",
    srcs = ["argmax_op.cc"],
    hdrs = ["argmax_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":reduction_utils",
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "ctc_op",
    srcs = ["ctc_loss_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/utils/ctc:ctc_loss_calculator_lib",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "bucketize_op",
    srcs = ["bucketize_op.cc"],
    hdrs = [
        "bucketize_op.h",
        "gpu_device_array.h",
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
    name = "argmin_op",
    srcs = ["argmin_op.cc"],
    hdrs = ["argmin_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":reduction_utils",
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "histogram_op",
    srcs = ["histogram_op.cc"],
    hdrs = [
        "histogram_op.h",
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
    name = "softplus_op",
    srcs = ["softplus_op.cc"],
    hdrs = ["softplus_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "cross_op",
    srcs = ["cross_op.cc"],
    hdrs = ["cross_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "softsign_op",
    srcs = ["softsign_op.cc"],
    hdrs = ["softsign_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "batchtospace_op",
    srcs = ["batchtospace_op.cc"],
    hdrs = [
        "spacetobatch_functor.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":dense_update_functor",
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "batch_norm_ops",
    srcs = [
        "batch_norm_op.cc",
        "batch_norm_op_gpu.cc",
    ],
    hdrs = [
        "batch_norm_op.h",
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
    name = "betainc_op",
    srcs = ["betainc_op.cc"],
    hdrs = [
        "betainc_op.h",
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
    name = "spacetodepth_op",
    srcs = ["spacetodepth_op.cc"],
    hdrs = [
        "spacetodepth_op.h",
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
    name = "bias_op",
    srcs = ["bias_op.cc"],
    hdrs = [
        "bias_op.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "bincount_op",
    srcs = ["bincount_op.cc"],
    hdrs = [
        "bincount_op.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "bitcast_op",
    srcs = ["bitcast_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "depthtospace_op",
    srcs = [
        "depthtospace_op.cc",
        "depthtospace_op_gpu.cc",
    ],
    hdrs = ["depthtospace_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "colorspace_ops",
    srcs = [
        "colorspace_op.cc",
        "colorspace_op_gpu.cc",
    ],
    hdrs = ["colorspace_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "inplace_ops",
    srcs = [
        "inplace_ops.cc",
        "inplace_ops_functor.cc",
    ],
    hdrs = [
        "inplace_ops_functor.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "broadcast_to_op",
    srcs = ["broadcast_to_op.cc"],
    hdrs = [
        "broadcast_to_op.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "cast_op",
    srcs = [
        "cast_op.cc",
        "cast_op_impl.cc",
    ],
    hdrs = [
        "cast_op.h",
        "cast_op_impl.h",
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
    name = "check_numerics_op",
    srcs = [
        "check_numerics_op.cc",
        "check_numerics_op_gpu.cc",
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
    name = "compare_and_bitpack",
    srcs = [
        "compare_and_bitpack_op.cc",
        "compare_and_bitpack_op_gpu.cc",
    ],
    hdrs = ["compare_and_bitpack_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "concat_v2_op",
    srcs = ["concat_op.cc"],
    hdrs = ["concat_lib.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "constant_ops",
    srcs = ["constant_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
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
    name = "data_format_ops",
    srcs = ["data_format_ops.cc"],
    hdrs = ["data_format_ops.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "debug_ops",
    srcs = ["debug_ops.cc"],
    hdrs = ["debug_ops.h"],
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
    name = "dense_update_functor",
    hdrs = ["dense_update_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "dense_update_op",
    srcs = ["dense_update_ops.cc"],
    hdrs = ["training_op_helpers.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":dense_update_functor",
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "dilation_op",
    srcs = [
        "dilation_ops.cc",
        "dilation_ops_gpu.cc",
    ],
    hdrs = [
        "cast_op.h",
        "dilation_ops.h",
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
    name = "diag_op",
    srcs = ["diag_op.cc"],
    hdrs = ["diag_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "dynamic_stitch_op",
    srcs = ["dynamic_stitch_op.cc"],
    hdrs = ["gpu_device_array.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "dynamic_partition_op",
    srcs = ["dynamic_partition_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "ops_util",
    srcs = ["ops_util.cc"],
    hdrs = ["ops_util.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "extract_image_patches_op",
    srcs = [
        "extract_image_patches_op.cc",
        "extract_image_patches_op_gpu.cc",
    ],
    hdrs = ["extract_image_patches_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":ops_util",
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "extract_volume_patches_op",
    srcs = [
        "extract_volume_patches_op.cc",
        "extract_volume_patches_op_gpu.cc",
    ],
    hdrs = ["extract_volume_patches_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":ops_util",
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fft_ops",
    srcs = ["fft_ops.cc"],
    hdrs = ["fft_ops.h"],
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
    hdrs = [
        "custom_fused_batch_norm_functor.h",
        "custom_fused_batch_norm_op.h",
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
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "gather_op",
    srcs = ["gather_op.cc"],
    hdrs = [
        "gather_functor.h",
        "gather_functor_batched.h",
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
    name = "gather_nd_op",
    srcs = [
        "gather_nd_op.cc",
        "gather_nd_op_gpu.cc",
    ],
    hdrs = [
        "gather_functor.h",
        "gather_nd_op.h",
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
    name = "in_topk_op",
    srcs = [
        "in_topk_op.cc",
        "in_topk_op_gpu.cc",
    ],
    hdrs = [
        "col_reduction_kernels.h",
        "full_reduction_kernels.h",
        "in_topk_op.h",
        "reduction_ops.h",
        "row_reduction_kernels.h",
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
    name = "instance_norm_op",
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
    name = "l2loss_op",
    srcs = ["l2loss_op.cc"],
    hdrs = ["full_reduction_kernels.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "layer_norm_op",
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
    name = "lrn_op",
    srcs = ["lrn_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "matrix_band_part_op",
    srcs = ["matrix_band_part_op.cc"],
    hdrs = ["matrix_band_part_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "matmul_op",
    srcs = ["matmul_op.cc"],
    hdrs = [
        "//itex/core/kernels/common:batch_matmul_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = ["//itex/core/kernels/common:fill_functor"],
    alwayslink = True,
)

itex_xpu_library(
    name = "quantized_matmul",
    srcs = ["quantized_matmul.cc"],
    hdrs = [
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

cc_library(
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
    name = "quantized_conv",
    srcs = ["quantized_conv_ops.cc"],
    hdrs = [
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
    name = "multinomial_op",
    srcs = [
        "multinomial_op.cc",
        "multinomial_op_gpu.cc",
    ],
    hdrs = [
        "multinomial_op.h",
        "random_op_gpu.h",
        "stateless_random_ops.h",
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
    name = "nextafter_op",
    srcs = ["nextafter_op.cc"],
    hdrs = [
        "nextafter_op.h",
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
    name = "quantize_and_dequantize_op",
    srcs = ["quantize_and_dequantize_op.cc"],
    hdrs = [
        "quantize_and_dequantize_op.h",
    ],
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
    name = "dequantize_reshape_op",
    srcs = ["dequantize_reshape_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "reduction_ops",
    srcs = [
        "reduction_ops_all.cc",
        "reduction_ops_any.cc",
        "reduction_ops_euclidean.cc",
        "reduction_ops_max.cc",
        "reduction_ops_mean.cc",
        "reduction_ops_min.cc",
        "reduction_ops_prod.cc",
        "reduction_ops_sum.cc",
    ],
    hdrs = [
        "col_reduction_kernels.h",
        "full_reduction_kernels.h",
        "reduction_ops.h",
        "reduction_ops_common.h",
        "row_reduction_kernels.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":reduction_utils",
        "//itex:core",
        "//itex/core/kernels/common:transpose_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "one_hot_op",
    srcs = ["one_hot_op.cc"],
    hdrs = ["one_hot_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "rnn_ops",
    srcs = [
        "rnn_ops.cc",
        "rnn_ops_gpu.cc",
    ],
    hdrs = [
        "col_reduction_kernels.h",
        "rnn_ops.h",
        "rnn_ops_gpu.h",
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
    name = "pack_op",
    srcs = [
        "concat_lib.cc",
        "pack_op.cc",
    ],
    hdrs = [
        "concat_lib.h",
        "gpu_device_array.h",
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
    name = "pad_op",
    srcs = ["pad_op.cc"],
    hdrs = ["pad_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "parameterized_truncated_normal_op",
    srcs = ["parameterized_truncated_normal_op.cc"],
    hdrs = ["parameterized_truncated_normal_op.h"],
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
    name = "avg_pooling_tmp",
    srcs = [
        "avgpooling_op.cc",
    ],
    hdrs = ["//itex/core/kernels/common:pooling_hdrs"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//itex:core",
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
        "maxpooling_op.h",
        "//itex/core/kernels/common:pooling_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = ["//itex:core"],
    alwayslink = True,
)

itex_xpu_library(
    name = "population_count_op",
    srcs = ["population_count_op.cc"],
    hdrs = ["population_count_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

cc_library(
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

cc_library(
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
    name = "random_op",
    srcs = [
        "random_op.cc",
        "random_op_gpu.cc",
    ],
    hdrs = [
        "random_op_gpu.h",
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
    srcs = [
        "fused_random_op.cc",
    ],
    hdrs = [
        "fused_random_op_gpu.h",
        "random_op_gpu.h",
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
    name = "relu_op",
    srcs = ["relu_op.cc"],
    hdrs = [
        "relu_op.h",
        "relu_op_functor.h",
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
    name = "resize_nearest_neighbor_op",
    srcs = ["resize_nearest_neighbor_op.cc"],
    hdrs = [
        "image_resizer_state.h",
        "resize_nearest_neighbor_op.h",
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
    name = "resource_variable_ops",
    srcs = ["resource_variable_ops.cc"],
    hdrs = [
        "dense_update_functor.h",
        "gather_functor.h",
        "gather_nd_op.h",
        "scatter_functor.h",
        "training_op_helpers.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "reverse_op",
    srcs = ["reverse_op.cc"],
    hdrs = [
        "reverse_op.h",
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
    name = "reverse_sequence_op",
    srcs = ["reverse_sequence_op.cc"],
    hdrs = [
        "reverse_sequence_op.h",
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
    name = "roll_op",
    srcs = ["roll_op.cc"],
    hdrs = [
        "roll_op.h",
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
    name = "scan_ops",
    srcs = [
        "scan_ops_1.cc",
        "scan_ops_2.cc",
        "scan_ops_3.cc",
    ],
    hdrs = [
        "scan_ops.h",
        "scan_ops_gpu.h",
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
    name = "scatter_nd_op",
    srcs = ["scatter_nd_op.cc"],
    hdrs = [
        "dense_update_functor.h",
        "inplace_ops_functor.h",
        "scatter_nd_op.h",
        "training_op_helpers.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "scatter_op",
    srcs = ["scatter_op.cc"],
    hdrs = [
        "dense_update_functor.h",
        "scatter_functor.h",
        "training_op_helpers.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "searchsorted_op",
    srcs = ["searchsorted_op.cc"],
    hdrs = ["searchsorted_op.h"],
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
    hdrs = [
        "ops_util.h",
        "slice_op.h",
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
    name = "snapshot_op",
    srcs = ["snapshot_op.cc"],
    hdrs = ["snapshot_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "softmax_op",
    srcs = ["softmax_op.cc"],
    hdrs = [
        "softmax_op_functor.h",
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
    name = "spacetobatch_op",
    srcs = [
        "spacetobatch_functor_gpu.cc",
        "spacetobatch_op.cc",
    ],
    hdrs = [
        "spacetobatch_functor.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "sparse_tensor_dense_mat_mul_op",
    srcs = ["sparse_tensor_dense_mat_mul_op.cc"],
    hdrs = ["sparse_tensor_dense_mat_mul_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "sparse_xent_op",
    srcs = [
        "sparse_xent_op.cc",
    ],
    hdrs = [
        "softmax_op_functor.h",
        "sparse_xent_op.h",
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
    name = "split_op",
    srcs = [
        "split_lib.cc",
        "split_op.cc",
    ],
    hdrs = [
        "gpu_device_array.h",
        "split_lib.h",
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
    name = "split_v_op",
    srcs = [
        "split_lib.cc",
        "split_v_op.cc",
    ],
    hdrs = [
        "gpu_device_array.h",
        "split_lib.h",
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
    name = "stateful_random_ops",
    srcs = [
        "stateful_random_ops.cc",
    ],
    hdrs = [
        "dense_update_functor.h",
        "random_op_gpu.h",
        "stateful_random_ops.h",
        "training_op_helpers.h",
        "//itex/core/kernels/common:random_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
        "//itex/core/utils/lib/random:guarded_philox_random",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "stateless_random_ops",
    srcs = [
        "stateless_random_ops.cc",
    ],
    hdrs = [
        "random_op_gpu.h",
        "stateless_random_ops.h",
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
    name = "stateless_random_gamma_op_v2",
    srcs = [
        "stateless_random_gamma_op_v2.cc",
    ],
    hdrs = [
        "random_op_gpu.h",
        "stateless_random_gamma_op_v2.h",
        "stateless_random_ops.h",
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
    name = "stateless_random_ops_v2",
    srcs = [
        "stateless_random_ops_v2.cc",
    ],
    hdrs = [
        "random_op_gpu.h",
        "stateless_random_ops.h",
        "stateless_random_ops_v2.h",
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
    name = "strided_slice_op",
    srcs = [
        "strided_slice_op.cc",
        "strided_slice_op_util.cc",
    ],
    hdrs = [
        "dense_update_functor.h",
        "inplace_ops_functor.h",
        "slice_op.h",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
        "strided_slice_op_util.h",
        "training_op_helpers.h",
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
    name = "transpose_op",
    srcs = [
        "transpose_op.cc",
    ],
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
    name = "tile_ops",
    srcs = [
        "tile_functor_gpu_bool.cc",
        "tile_functor_gpu_complex128.cc",
        "tile_functor_gpu_complex64.cc",
        "tile_functor_gpu_double.cc",
        "tile_functor_gpu_float.cc",
        "tile_functor_gpu_half.cc",
        "tile_functor_gpu_int16.cc",
        "tile_functor_gpu_int32.cc",
        "tile_functor_gpu_int64.cc",
        "tile_ops.cc",
        "tile_ops_gpu_impl_1.cc",
        "tile_ops_gpu_impl_2.cc",
        "tile_ops_gpu_impl_3.cc",
        "tile_ops_gpu_impl_4.cc",
        "tile_ops_gpu_impl_5.cc",
        "tile_ops_gpu_impl_6.cc",
        "tile_ops_gpu_impl_7.cc",
        "tile_ops_gpu_impl_8.cc",
    ],
    hdrs = [
        "tile_functor.h",
        "tile_functor_gpu.h",
        "tile_ops_gpu_impl.h",
        "tile_ops_impl.h",
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
    name = "training_ops",
    srcs = [
        "training_op_ada_max.cc",
        "training_op_adadelta.cc",
        "training_op_adagrad.cc",
        "training_op_adagradv2.cc",
        "training_op_adam.cc",
        "training_op_add_sign.cc",
        "training_op_ftrl.cc",
        "training_op_gradient_descent.cc",
        "training_op_helpers.cc",
        "training_op_keras_momentum.cc",
        "training_op_momentum.cc",
        "training_op_power_sign.cc",
        "training_op_proximal_adagrad.cc",
        "training_op_rmsprop.cc",
    ],
    hdrs = [
        "dense_update_functor.h",
        "training_op_helpers.h",
        "training_ops.h",
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
    name = "tridiagonal_matmul_op",
    srcs = [
        "tridiagonal_matmul_op.cc",
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
    name = "segment_reduction_ops",
    srcs = ["segment_reduction_ops.cc"],
    hdrs = [
        "segment_reduction_ops.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "topk_functor_header",
    hdrs = ["topk_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "topk_op",
    srcs = [
        "topk_op.cc",
        "topk_op_gpu.cc",
    ],
    hdrs = ["topk_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "sequence_ops",
    srcs = ["sequence_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "resize_bilinear_ops",
    srcs = ["resize_bilinear_op.cc"],
    hdrs = [
        "cast_op.h",
        "image_resizer_state.h",
        "resize_bilinear_op.h",
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
    name = "select_op",
    srcs = ["select_op.cc"],
    hdrs = ["cwise_op.h"],
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
    name = "unary_and_binary_kernels",
    srcs = ["unary_and_binary_ops"],
    hdrs = ["cwise_op_clip.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:cwise_ops_lib",
    ],
    alwayslink = True,
)

filegroup(
    name = "unary_and_binary_ops",
    srcs = [
        "cwise_op_abs.cc",
        "cwise_op_acos.cc",
        "cwise_op_acosh.cc",
        "cwise_op_add.cc",
        "cwise_op_approximate_equal.cc",
        "cwise_op_arg.cc",
        "cwise_op_asin.cc",
        "cwise_op_asinh.cc",
        "cwise_op_atan.cc",
        "cwise_op_atan2.cc",
        "cwise_op_atanh.cc",
        "cwise_op_bitwise_and.cc",
        "cwise_op_bitwise_or.cc",
        "cwise_op_bitwise_xor.cc",
        "cwise_op_ceil.cc",
        "cwise_op_clip.cc",
        "cwise_op_complex.cc",
        "cwise_op_conj.cc",
        "cwise_op_cos.cc",
        "cwise_op_cosh.cc",
        "cwise_op_digamma.cc",
        "cwise_op_div.cc",
        "cwise_op_equal_to.cc",
        "cwise_op_erf.cc",
        "cwise_op_erfc.cc",
        "cwise_op_exp.cc",
        "cwise_op_expm1.cc",
        "cwise_op_floor.cc",
        "cwise_op_floor_div.cc",
        "cwise_op_floor_mod.cc",
        "cwise_op_greater.cc",
        "cwise_op_greater_equal.cc",
        "cwise_op_igammas.cc",
        "cwise_op_imag.cc",
        "cwise_op_inverse.cc",
        "cwise_op_invert.cc",
        "cwise_op_isfinite.cc",
        "cwise_op_isinf.cc",
        "cwise_op_isnan.cc",
        "cwise_op_left_shift.cc",
        "cwise_op_less.cc",
        "cwise_op_less_equal.cc",
        "cwise_op_lgamma.cc",
        "cwise_op_log.cc",
        "cwise_op_log1p.cc",
        "cwise_op_logical_and.cc",
        "cwise_op_logical_not.cc",
        "cwise_op_logical_or.cc",
        "cwise_op_maximum.cc",
        "cwise_op_minimum.cc",
        "cwise_op_mod.cc",
        "cwise_op_mul.cc",
        "cwise_op_ndtri.cc",
        "cwise_op_neg.cc",
        "cwise_op_not_equal_to.cc",
        "cwise_op_pow.cc",
        "cwise_op_random_grad.cc",
        "cwise_op_real.cc",
        "cwise_op_reciprocal.cc",
        "cwise_op_right_shift.cc",
        "cwise_op_rint.cc",
        "cwise_op_round.cc",
        "cwise_op_rsqrt.cc",
        "cwise_op_sigmoid.cc",
        "cwise_op_sign.cc",
        "cwise_op_sin.cc",
        "cwise_op_sinh.cc",
        "cwise_op_sqrt.cc",
        "cwise_op_square.cc",
        "cwise_op_squared_difference.cc",
        "cwise_op_sub.cc",
        "cwise_op_tan.cc",
        "cwise_op_tanh.cc",
        "cwise_op_xdivy.cc",
        "cwise_op_xlog1py.cc",
        "cwise_op_xlogy.cc",
        "cwise_op_zeta.cc",
    ],
    visibility = ["//visibility:public"],
)

itex_xpu_library(
    name = "unique_op",
    srcs = ["unique_op.cc"],
    hdrs = [
        "topk_op.h",
        "unique_op.h",
        "unique_op_helpers.h",
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
    name = "unpack_op",
    srcs = [
        "split_lib.cc",
        "unpack_op.cc",
    ],
    hdrs = [
        "gpu_device_array.h",
        "split_lib.h",
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
    name = "where_op",
    srcs = ["where_op.cc"],
    hdrs = [
        "scan_ops_gpu.h",
        "where_op.h",
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
    name = "xent_op",
    srcs = ["xent_op.cc"],
    hdrs = [
        "softmax_op_functor.h",
        "xent_op.h",
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
    name = "tensorarray_ops",
    srcs = [
        "concat_lib.cc",
        "split_lib.cc",
        "tensor_array.cc",
        "tensor_array_ops.cc",
    ],
    hdrs = [
        "aggregate_ops.h",
        "concat_lib.h",
        "gpu_device_array.h",
        "split_lib.h",
        "tensor_array.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

GPU_KERNELS = [
    ":aggregate_ops",
    ":argmax_op",
    ":argmin_op",
    ":batchtospace_op",
    ":batch_norm_ops",
    ":betainc_op",
    ":spacetodepth_op",
    ":bucketize_op",
    ":bincount_op",
    ":bitcast_op",
    ":broadcast_to_op",
    ":cast_op",
    ":check_numerics_op",
    ":colorspace_ops",
    ":compare_and_bitpack",
    ":concat_v2_op",
    ":ctc_op",
    ":bias_op",
    ":constant_ops",
    ":cross_op",
    ":conv_ops",
    ":data_format_ops",
    ":debug_ops",
    ":dense_update_op",
    ":depthtospace_op",
    ":dequantize_op",
    ":dequantize_reshape_op",
    ":dilation_op",
    ":diag_op",
    ":dynamic_stitch_op",
    # TODO(itex): Temporarily disable DynamicPartition op as its gpu implementation needs refactor.
    # ":dynamic_partition_op"
    ":extract_image_patches_op",
    ":extract_volume_patches_op",
    ":fft_ops",
    ":fused_batch_norm_op",
    ":fused_binary_op",
    ":fused_random_op",
    ":gather_op",
    ":gather_nd_op",
    ":histogram_op",
    ":inplace_ops",
    ":in_topk_op",
    ":instance_norm_op",
    ":topk_op",
    ":l2loss_op",
    ":layer_norm_op",
    ":lrn_op",
    ":matrix_band_part_op",
    ":matmul_op",
    ":multinomial_op",
    ":nextafter_op",
    ":one_hot_op",
    ":pack_op",
    ":pad_op",
    ":parameterized_truncated_normal_op",
    ":pooling_ops",
    ":population_count_op",
    ":quantize_op",
    ":quantize_and_dequantize_op",
    ":quantized_concat_op",
    ":quantized_conv",
    ":quantized_matmul",
    ":quantized_reshape_op",
    ":random_op",
    ":reduction_ops",
    ":resize_nearest_neighbor_op",
    ":resource_variable_ops",
    ":relu_op",
    ":resize_bilinear_ops",
    ":reverse_op",
    ":reverse_sequence_op",
    ":roll_op",
    ":rnn_ops",
    ":scan_ops",
    ":scatter_nd_op",
    ":scatter_op",
    ":searchsorted_op",
    ":segment_reduction_ops",
    ":select_op",
    ":sequence_ops",
    ":slice_op",
    ":snapshot_op",
    ":softmax_op",
    ":softplus_op",
    ":softsign_op",
    ":spacetobatch_op",
    ":sparse_tensor_dense_mat_mul_op",
    ":sparse_xent_op",
    ":split_op",
    ":split_v_op",
    ":stateful_random_ops",
    ":stateless_random_ops",
    ":stateless_random_ops_v2",
    ":stateless_random_gamma_op_v2",
    ":strided_slice_op",
    ":tile_ops",
    ":training_ops",
    ":transpose_op",
    ":tridiagonal_matmul_op",
    ":unary_and_binary_kernels",
    ":unique_op",
    ":unpack_op",
    ":where_op",
    ":xent_op",
    ":tensorarray_ops",
]

itex_xpu_library(
    name = "gpu_kernel_impl",
    srcs = ["gpu_kernel_init.cc"],
    hdrs = [
        "gpu_kernel_init.h",
        "//itex/core/devices/gpu:gpu_device_plugin.h",
        "//itex/core/utils:padding.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/legacy:legacy_kernels",
        "//itex/core/kernels/onednn_graph:onednn_graph_op",
        "//itex/core/kernels/onednn:onednn_kernels",
        "//itex/core/kernels/gpu/linalg:linalg",
        "//itex/core/kernels/gpu/special_math:special_math_op",
        "//itex/core/kernels/gpu/image:crop_and_resize_op",
        "//itex/core/kernels/gpu/image:image",
    ] + GPU_KERNELS,
    alwayslink = True,
)
