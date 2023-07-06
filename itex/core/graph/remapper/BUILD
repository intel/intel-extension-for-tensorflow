load("//itex:itex.bzl", "cc_library")

cc_library(
    name = "remapper",
    srcs = [
        "activation_pattern.cc",
        "batch_matmul_pattern.cc",
        "cast_fused_matmul_cast_pattern.cc",
        "cast_matmul_cast_pattern.cc",
        "conv_backprop_input_pattern.cc",
        "fusion.cc",
        "gru_pattern.cc",
        "instance_norm_pattern.cc",
        "layer_norm_pattern.cc",
        "pad_conv3d_with_cast_pattern.cc",
        "pad_conv_pattern.cc",
        "remapper.cc",
        "resize_image_pattern.cc",
        "rmsprop_pattern.cc",
    ],
    hdrs = [
        "constant_names.h",
        "fusion.h",
        "remapper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/graph:optimizer_config",
        "//itex/core/graph/utils:graph_common_utils",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:layout_utils",
        "//itex/core/graph/utils:pattern_utils",
        "//itex/core/graph/utils:symbolic_shapes",
    ],
    alwayslink = True,
)
