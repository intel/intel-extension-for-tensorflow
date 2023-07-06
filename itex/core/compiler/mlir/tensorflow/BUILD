package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorflow_types",
    hdrs = [
        "ir/tf_dialect.h",
        "ir/tf_types.h",
    ],
    textual_hdrs = [
        "ir/tf_types.def",
    ],
    deps = [
        "//itex/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "convert_type",
    srcs = [
        "utils/convert_type.cc",
    ],
    hdrs = [
        "utils/convert_type.h",
    ],
    textual_hdrs = [
        "ir/tf_types.def",
    ],
    deps = [
        ":tensorflow_types",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "error_util",
    srcs = ["utils/error_util.cc"],
    hdrs = ["utils/error_util.h"],
    # visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//itex/core/utils:common_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
