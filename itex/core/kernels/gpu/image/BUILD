load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

# List all kernels here.
itex_xpu_library(
    name = "mirror_pad_op",
    srcs = ["mirror_pad_op.cc"],
    hdrs = ["mirror_pad_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "adjust_contrast_op",
    srcs = ["adjust_contrast_op.cc"],
    hdrs = ["adjust_contrast_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "adjust_saturation_op",
    srcs = ["adjust_saturation_op.cc"],
    hdrs = ["adjust_saturation_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "image_ops",
    srcs = ["image_ops.cc"],
    hdrs = ["image_ops.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "adjust_contrastv2_op",
    srcs = ["adjust_contrastv2_op.cc"],
    hdrs = ["adjust_contrastv2_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "adjust_hue_op",
    srcs = ["adjust_hue_op.cc"],
    hdrs = ["adjust_hue_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "non_max_suppression_op",
    srcs = [
        "non_max_suppression_op.cc",
        "non_max_suppression_op_gpu.cc",
    ],
    hdrs = [
        "non_max_suppression_op.h",
        "non_max_suppression_op_gpu.h",
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
    name = "combined_non_max_suppression_op",
    srcs = [
        "combined_non_max_suppression_op.cc",
        "combined_non_max_suppression_op_gpu.cc",
    ],
    hdrs = [
        "combined_non_max_suppression_op.h",
        "combined_non_max_suppression_op_gpu.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/gpu:topk_functor_header",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "generate_bounding_box_proposals_op",
    srcs = ["generate_box_proposals_op.cc"],
    hdrs = ["non_max_suppression_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/gpu:topk_functor_header",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "image",
    visibility = ["//visibility:public"],
    deps = [
        ":adjust_contrast_op",
        ":adjust_contrastv2_op",
        ":adjust_hue_op",
        ":adjust_saturation_op",
        ":combined_non_max_suppression_op",
        ":generate_bounding_box_proposals_op",
        ":image_ops",
        ":mirror_pad_op",
        ":non_max_suppression_op",
    ],
)

itex_xpu_library(
    name = "crop_and_resize_op",
    srcs = ["crop_and_resize_op.cc"],
    hdrs = ["crop_and_resize_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/gpu:cast_op",
    ],
    alwayslink = True,
)
